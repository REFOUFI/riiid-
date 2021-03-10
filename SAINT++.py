#!/usr/bin/env python
# coding: utf-8

# # SAINT+ Encoder Decoder Model

# In[1]:

import datetime
import warnings
warnings.filterwarnings("ignore")

import gc, sys, os
import random, math
from tqdm import tqdm

import numpy as np
import pandas as pd

from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

import seaborn as sns
import matplotlib.pyplot as plt


# In[2]:


import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn_utils
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
#import pytorch_warmup as warmup

import seaborn as sns
sns.set()
DEFAULT_FIG_WIDTH = 20
sns.set_context("paper", font_scale=1.2) 


# In[3]:


DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu') # if IS_TPU == False else xm.xla_device()
print('Running on device: {}'.format(DEVICE))


# In[4]:


def seed_everything(s):
    random.seed(s)
    os.environ['PYTHONHASHSEED'] = str(s)
    np.random.seed(s)
    # Torch
    torch.manual_seed(s)
    torch.cuda.manual_seed(s)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)

seed = 42
seed_everything(seed)
#os.environ["CUDA_LAUNCH_BLOCKING"] = "1"


# In[5]:
current_date_and_time = datetime.datetime.now()
current_date_and_time_string = str(current_date_and_time)
extension = ".txt"

file_name = 'train'+current_date_and_time_string + extension
file = open(file_name, 'w')
file.close()
def write_in_file(filename,text):
    file = open(filename, "a")
    file.write(text+"\n")
    file.close()



HOME =  "./"
DATA_HOME = "./data/"
MODEL_NAME = "SAINT+-v0"
MODEL_PATH = HOME + MODEL_NAME
STAGE = "stage1"
MODEL_BEST = 'model_best.pt'
FOLD = 1

if not os.path.exists(MODEL_PATH):
    os.makedirs(MODEL_PATH)
    
ROW_ID = "row_id"  
TIMESTAMP = "timestamp" 
USER_ID = "user_id"
CONTENT_ID = "content_id"
CONTENT_TYPE_ID = "content_type_id"
TARGET = "answered_correctly"
ELAPSED_TIME = "question_elapsed_time"
TASK_CONTAINER_ID = "task_container_id"
PART = "part"
LAG_TIME = "lag_time"
RESID = "resid"
QUESTION_CORRECTNESS = "question_correctness"
QUESTION_SEEN = "question_seen"


# In[6]:


train_df=pd.read_pickle("saint_data.pkl")


# In[7]:


def get_train_val_idxs(TRAIN_SIZE, VAL_SIZE):
    train_idxs_user = []
    train_idxs = []
    val_idxs_user = []
    val_idxs = []
    NEW_USER_FRAC = 1/4 # fraction of new users in 
    #np.random.seed(42)
    
    # create df with user_ids and indices
    df = train_df[['user_id']]

    df['index'] = df.index.values.astype(np.uint32)
    user_first_index = df.groupby('user_id')['index'].first()
    user_id_index = df.groupby('user_id')['index'].apply(np.array)
    
    shuffle = user_id_index.sample(user_id_index.size, random_state=42)
    
    # iterate over users in random order
    for index,indices in tqdm(zip(shuffle.index,shuffle)):
        if len(train_idxs) > TRAIN_SIZE:
            break

        # fill validation data
        if len(val_idxs) < VAL_SIZE:
            # add new user
            if np.random.rand() < NEW_USER_FRAC:
                val_idxs_user += list([index]*len(indices))
                val_idxs += list(indices)
            # randomly split user between train and val otherwise
            else:
                offset = np.random.randint(0, indices.size)
                train_idxs_user += list([index]*offset)
                val_idxs_user += list([index]*(len(indices)-offset))
                train_idxs += list(indices[:offset])
                val_idxs += list(indices[offset:])
        else:
            train_idxs_user += list([index]*len(indices))
            train_idxs += list(indices)
    
    
    del df,shuffle,user_id_index
    gc.collect()
        
    return train_idxs,train_idxs_user,val_idxs,val_idxs_user,user_first_index

train_idxs,train_idxs_user,val_idxs,val_idxs_user,user_first_index = get_train_val_idxs(int(50e6), 2.5e6)
print(f'len train_idxs: {len(train_idxs)}, len validation_idxs: {len(val_idxs)}')


# ## Preprocess

# In[8]:


# Index by user_id
#train_df.sort_values([USER_ID, TIMESTAMP], ascending=[True, True], inplace=True) # Already sorted
#train_df = train_df.reset_index(drop = True)
train_group = train_df.groupby(USER_ID).apply(lambda r: (r[CONTENT_ID].values,r[PART].values, r[TARGET].values,
                                                         r[ELAPSED_TIME].values,r[LAG_TIME].values,r[RESID].values))
del train_df
gc.collect()


# ## SAINT+ Dataset

# In[9]:


class SAINTDataset(Dataset):
    def __init__(self, group, subset_idx,subset_idx_user,user_start_index, max_seq=100):
        super(SAINTDataset, self).__init__()
        self.max_seq = max_seq
        self.samples = group
        self.subset_idx = subset_idx
        self.subset_idx_user = subset_idx_user
        self.user_start_index = user_start_index
        
        # self.user_ids = [x for x in group.index]
        #self.user_ids = []
        #for user_id in group.index:
        #    self.user_ids.append(user_id) # user_ids indexes
            
    def __len__(self):
        return len(self.subset_idx)

    def __getitem__(self, index):
        
        real_index=self.subset_idx[index]
        user_id= self.subset_idx_user[index]
        
        index_in_sample = real_index - self.user_start_index[user_id]
        
            
        q_, pa_, qa_,el_time_,la_time_,resid_ = self.samples[user_id] # Pick full sequence for user
        

        q = np.zeros(self.max_seq, dtype=int)
        pa = np.zeros(self.max_seq, dtype=int)
        qa = np.zeros(self.max_seq, dtype=int)
        el_time = np.zeros(self.max_seq, dtype=np.float32)
        la_time = np.zeros(self.max_seq, dtype=int)
        resid = np.zeros(self.max_seq, dtype=np.float32)
        pad = np.ones(self.max_seq, dtype=bool)
        
        seq_len=index_in_sample+1

        if seq_len >= self.max_seq:
            #random_start_index = np.random.randint(seq_len - self.max_seq)
            q[:] = q_[seq_len-self.max_seq:seq_len] # Pick 100 questions from a random index
            pa[:] = pa_[seq_len-self.max_seq:seq_len]
            qa[:] = qa_[seq_len-self.max_seq:seq_len] # Pick 100 answers from a random index
            el_time[:] = el_time_[seq_len-self.max_seq:seq_len]
            la_time[:] = la_time_[seq_len-self.max_seq:seq_len]
            resid[:] = resid_[seq_len-self.max_seq:seq_len]
            pad[:]=False
        else:
            q[-seq_len:] = q_[:seq_len] # Pick last N question with zero padding
            pa[-seq_len:] = pa_[:seq_len] # Pick last N answers with zero padding 
            qa[-seq_len:] = qa_[:seq_len] # Pick last N answers with zero padding  
            el_time[-seq_len:] = el_time_[:seq_len] # Pick last N answers with zero padding
            la_time[-seq_len:] = la_time_[:seq_len] # Pick last N answers with zero padding
            resid[-seq_len:] = resid_[:seq_len]
            pad[-seq_len:]=False
        
        # x = np.zeros(self.max_seq-1, dtype=int)
        excercise = q[1:].copy() # 0 to 98
        part= pa[1:]
        prior_correctness = qa[:-1].copy() # Ignore first item 1 to 99
        prior_elapse_time= np.nan_to_num(el_time[:-1]) 
        prior_lag_time= la_time[:-1]
        residual= resid[:-1]
        padding=pad[1:]
    
        label = qa[1:] # Ignore first item 1 to 99

        return excercise, part, prior_correctness, prior_elapse_time, prior_lag_time, residual, padding, label


# ## Define model

# In[10]:


import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer,TransformerDecoderLayer,TransformerDecoder

def future_mask(seq_length,device):
    future_mask = np.triu(np.ones((seq_length, seq_length)), k=1).astype('bool')
    return torch.tensor(future_mask,dtype=torch.bool,device=device)

class SAINTModel(nn.Module):
    def __init__(self, max_seq=100, nhead=8, embed_dim=128,nlayers=2,dropout=0.1):
        super(SAINTModel, self).__init__()
        
        self.max_seq = max_seq
        self.embed_dim = embed_dim

        #self.embedding = nn.Embedding(2*n_skill+1, embed_dim)
        self.pos_embedding = nn.Embedding(max_seq-1, embed_dim)
        self.exercise_embeddings = nn.Embedding(13523+1, embed_dim)
        self.part_embedding = nn.Embedding(7+1, embed_dim)
        self.prior_elapsed_time_embedding= nn.Linear(1,embed_dim, False)
        #self.residual_embedding= nn.Linear(1,embed_dim, False)
        self.prior_lag_time_embedding = nn.Embedding(1442, embedding_dim=embed_dim)
        self.prior_correctness_embedding = nn.Embedding(3, embedding_dim=embed_dim)
        
        encoder_layers = TransformerEncoderLayer(d_model=embed_dim, nhead=nhead, dim_feedforward=embed_dim, dropout=dropout, activation='relu')
        decoder_layers = nn.TransformerDecoderLayer(d_model=embed_dim, nhead=nhead, dim_feedforward=embed_dim, dropout=dropout, activation='relu')
        self.transformer_encoder = TransformerEncoder(encoder_layer=encoder_layers, num_layers=nlayers)
        self.transformer_decoder = TransformerDecoder(decoder_layer=decoder_layers, num_layers=nlayers)

        self.linear = nn.Linear(embed_dim, 1)
    
    def forward(self, excercise, part, prior_correctness, prior_elapse_time, prior_lag_time, residual, mask):
        
        device = excercise.device 
        # Encoder Input
        pos_inputs = torch.arange(excercise.size(1),device=device).unsqueeze(0)
        pos_inp = self.pos_embedding(pos_inputs)
        
        inputs_embedding = self.exercise_embeddings(excercise) + pos_inp+ self.part_embedding(part)
        
        
        # Decoder Input
        pos_target = torch.arange(prior_correctness.size(1),device=device).unsqueeze(0)
        pos_tar = self.pos_embedding(pos_target)
        
        elased_time = self.prior_elapsed_time_embedding(prior_elapse_time.unsqueeze(2))
        
        #residual_embedding = self.residual_embedding(residual.unsqueeze(2))
        
        target_embedding= self.prior_correctness_embedding(prior_correctness)+pos_tar+elased_time+self.prior_lag_time_embedding(prior_lag_time)# (N, S, E)

        # Transpose
        inputs_embedding = inputs_embedding.transpose(0, 1) # (S, N, E)
        target_embedding=target_embedding.transpose(0, 1)
               
        # Masking
        target_mask = future_mask(target_embedding.size(0),device)
        
        # Forward 
        enc_output = self.transformer_encoder(inputs_embedding,src_key_padding_mask=mask)
        
        output = self.transformer_decoder(target_embedding, enc_output,tgt_mask=target_mask,memory_key_padding_mask=mask)
        
        #output = self.ffn(output)
        
        output = self.linear(output)
        
        output=output.transpose(0, 1)
        
        #print("output:"+str(output.size()))
        
        return output.squeeze(2)


# ### Train Epoch

# In[11]:


def train_epoch(model,epoch,train_iterator, optim, criterion,device="cpu"):#,lr_scheduler,warmup_scheduler
    
    model.train()

    train_loss = []
    num_corrects = 0
    num_total = 0
    labels = []
    outs = []

    tbar = tqdm(train_iterator)
    for item in tbar: #:train_iterator
        exercise = torch.tensor(item[0],dtype=torch.long,device=device)
        part = torch.tensor(item[1],dtype=torch.long,device=device)
        prior_correctness = torch.tensor(item[2],dtype=torch.long,device=device)
        prior_elapse_time = torch.tensor(item[3],dtype=torch.float32,device=device)
        prior_lag_time = torch.tensor(item[4],dtype=torch.long,device=device)
        residual = torch.tensor(item[5],dtype=torch.float32,device=device)
        padding= torch.tensor(item[6],dtype=torch.bool,device=device)
        label = torch.tensor(item[7],dtype=torch.float32,device=device)
        #lr_scheduler.step(epoch-1)
        #warmup_scheduler.dampen()
        
        optim.zero_grad()
        output = model(exercise, part,prior_correctness,prior_elapse_time,prior_lag_time,residual,padding)
        loss = criterion(output, label)
        loss.backward()
        optim.step()
        train_loss.append(loss.item())

        output = output[:, -1]
        label = label[:, -1] 
        pred = (torch.sigmoid(output) >= 0.5).long()
        
        num_corrects += (pred == label).sum().item()
        num_total += len(label)

        labels.extend(label.view(-1).data.detach().cpu().numpy())
        #outs.extend(output.view(-1).data.cpu().numpy())
        outs.extend(torch.sigmoid(output).view(-1).data.detach().cpu().numpy())

        tbar.set_description('loss - {:.4f}'.format(loss))
    
    acc = num_corrects / num_total
    auc = roc_auc_score(labels, outs)
    loss = np.mean(train_loss)

    return loss, acc, auc


# ### Valid Epoch

# In[12]:


def valid_epoch(model, valid_iterator, criterion, device="cpu"):
    model.eval()

    valid_loss = []
    num_corrects = 0
    num_total = 0
    labels = []
    outs = []

    tbar = tqdm(valid_iterator)
    for item in tbar: #  valid_iterator:
        exercise = torch.tensor(item[0],dtype=torch.long,device=device)
        part = torch.tensor(item[1],dtype=torch.long,device=device)
        prior_correctness = torch.tensor(item[2],dtype=torch.long,device=device)
        prior_elapse_time = torch.tensor(item[3],dtype=torch.float32,device=device)
        prior_lag_time = torch.tensor(item[4],dtype=torch.long,device=device)
        residual = torch.tensor(item[5],dtype=torch.float32,device=device)
        padding= torch.tensor(item[6],dtype=torch.bool,device=device)
        label = torch.tensor(item[7],dtype=torch.float32,device=device)
        

        with torch.no_grad():
            output = model(exercise, part,prior_correctness,prior_elapse_time,prior_lag_time,residual,padding)
        loss = criterion(output, label)
        valid_loss.append(loss.item())

        output = output[:, -1] # (BS, 1)
        label = label[:, -1] 
        pred = (torch.sigmoid(output) >= 0.5).long()
        
        num_corrects += (pred == label).sum().item()
        num_total += len(label)

        labels.extend(label.view(-1).data.detach().cpu().numpy())
        #outs.extend(output.view(-1).data.cpu().numpy())
        outs.extend(torch.sigmoid(output).view(-1).data.detach().cpu().numpy())
        
        tbar.set_description('loss - {:.4f}'.format(loss))

    acc = num_corrects / num_total
    auc = roc_auc_score(labels, outs)
    loss = np.mean(valid_loss)

    return loss, acc, auc


# ## Prepare To Train

# In[13]:


class conf:
    METRIC_ = "max"
    WORKERS = 16 # 0
    BATCH_SIZE = 1024
    lr = 1e-3
    D_MODEL = 512
    DROPOUT = 0.1
    N_HEAD = 8
    N_ENC_DEC_LAYERS=2

    if torch.cuda.is_available():
        map_location=lambda storage, loc: storage.cuda()
    else:
        map_location='cpu'


# ### Prepare Train and Valid Dataset

# In[14]:


train_dataset = SAINTDataset(train_group,subset_idx=train_idxs,subset_idx_user=train_idxs_user,user_start_index=user_first_index)
train_dataloader = DataLoader(train_dataset, batch_size=conf.BATCH_SIZE, shuffle=True, num_workers=conf.WORKERS)

valid_dataset = SAINTDataset(train_group,subset_idx=val_idxs,subset_idx_user=val_idxs_user,user_start_index=user_first_index)
valid_dataloader = DataLoader(valid_dataset, batch_size=conf.BATCH_SIZE, shuffle=False, num_workers=conf.WORKERS)

del train_group,train_idxs,train_idxs_user,val_idxs,val_idxs_user,user_first_index
gc.collect()

#item = train_dataset.__getitem__(1888)

#print("exercise", len(item[0]), item[0])
#print("residual", len(item[5]), item[5])
#print("padding", len(item[6]), item[6])
#print("label", len(item[7]), item[7])


# ### Prepare Model 

# In[15]:


device = DEVICE

model = SAINTModel(embed_dim=conf.D_MODEL,nhead=conf.N_HEAD,nlayers=conf.N_ENC_DEC_LAYERS,dropout=conf.DROPOUT)
optimizer = torch.optim.Adam(model.parameters(), lr=conf.lr,betas=(0.9,0.999),eps=1e-8)
#optimizer = torch.optim.AdamW(model.parameters(), lr=conf.lr, betas=(0.9, 0.999),eps=1e-8)
criterion = nn.BCEWithLogitsLoss()
if torch.cuda.device_count() > 1:
    print("Let's use", torch.cuda.device_count(), "GPUs!")
    # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
    model = nn.DataParallel(model)

model.to(device)
criterion.to(device)

def weights_init(m):
    if isinstance(m, nn.Linear):
        #print('baba')
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
            

model.apply(weights_init)


# ### Training 

# In[16]:


epochs = 48
auc_max = -np.inf
history = []
num_steps = len(train_dataloader) * epochs
#lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_steps)
#warmup_scheduler = warmup.LinearWarmup(optimizer, warmup_period=4000)
snapshot_path = "%s/fold%d/%s/snapshots" % (MODEL_PATH, FOLD, STAGE)
if not os.path.exists(snapshot_path):
    os.makedirs(snapshot_path)

print("Stage:", STAGE, "fold:", FOLD, "on:", DEVICE, "workers:", conf.WORKERS, "batch size:", conf.BATCH_SIZE, "metric_:", conf.METRIC_, 
      "train dataset:", len(train_dataset), "valid dataset:", len(valid_dataset))

for epoch in range(1, epochs+1):
    train_loss, train_acc, train_auc = train_epoch(model,epoch,train_dataloader, optimizer, criterion,device)#lr_scheduler,warmup_scheduler,
    train_result="\nEpoch#{}, train_loss - {:.2f} acc - {:.4f} auc - {:.4f}".format(epoch, train_loss, train_acc, train_auc)
    print(train_result)
    write_in_file(file_name,train_result)
    valid_loss, valid_acc, valid_auc = valid_epoch(model, valid_dataloader, criterion, device)
    valid_result="Epoch#{}, valid_loss - {:.2f} acc - {:.4f} auc - {:.4f}".format(epoch, valid_loss, valid_acc, valid_auc)
    print(valid_result)
    write_in_file(file_name,valid_result)
    lr = optimizer.param_groups[0]['lr']
    history.append({"epoch":epoch, "lr": lr, **{"train_auc": train_auc, "train_acc": train_acc}, **{"valid_auc": valid_auc, "valid_acc": valid_acc}})
    if valid_auc > auc_max:
        improvement="Epoch#%s, valid loss %.4f, Metric loss improved from %.4f to %.4f, saving model ..." % (epoch, valid_loss, auc_max, valid_auc)
        print(improvement)
        write_in_file(file_name,improvement)
        auc_max = valid_auc
        torch.save(model.state_dict(), os.path.join(snapshot_path, MODEL_BEST))

if history:
    metric = "auc"
    # Plot training history
    history_pd = pd.DataFrame(history[1:]).set_index("epoch")
    train_history_pd = history_pd[[c for c in history_pd.columns if "train_" in c]]
    valid_history_pd = history_pd[[c for c in history_pd.columns if "valid_" in c]]
    lr_history_pd = history_pd[[c for c in history_pd.columns if "lr" in c]]
    fig, ax = plt.subplots(1,2, figsize=(DEFAULT_FIG_WIDTH, 6))
    t_epoch = train_history_pd["train_%s" % metric].argmin() if conf.METRIC_ == "min" else train_history_pd["train_%s" % metric].argmax()
    v_epoch = valid_history_pd["valid_%s" % metric].argmin() if conf.METRIC_ == "min" else valid_history_pd["valid_%s" % metric].argmax()
    d = train_history_pd.plot(kind="line", ax=ax[0], title="Epoch: %d, Train: %.3f" % (t_epoch, train_history_pd.iloc[t_epoch,:]["train_%s" % metric]))
    d = lr_history_pd.plot(kind="line", ax=ax[0], secondary_y=True)
    d = valid_history_pd.plot(kind="line", ax=ax[1], title="Epoch: %d, Valid: %.3f" % (v_epoch, valid_history_pd.iloc[v_epoch,:]["valid_%s" % metric]))
    d = lr_history_pd.plot(kind="line", ax=ax[1], secondary_y=True)
    plt.savefig("%s/train.png" % snapshot_path, bbox_inches='tight')
    plt.show()


# In[ ]:




