# riiid- competition    kaggle 

kaggle competition-riiid 

about the compettion : 

https://www.kaggle.com/c/riiid-test-answer-prediction


this was our first competetion ever , either for my teammate [@ahmed3991](https://github.com/ahmed3991) and I 

this work is from our  first kaggle compettion , it's an lgbm model 

we  also develped, an other approach using tranformers in pytorch 

the fair challenge in RIIID COMPETETION was to implent a good piplinde for the test data , wich require a lot of software engerring , with a hidden set of testing data , updating the the new users features without knowing if the sequence is a lecture or question was realy challenging .


during this competetion we  submitted more than 100 times , around half with lgbm models , and half with a pytorch transformers model
a lot of submmision scored errors , either the  ones with a valid output 


the first notebook(lgbm-COPY 1) is the model developpemnt , implemnted ,trained and validated on  local machine (macboook )

lgbm copy2 is  just a copy more detailed and explained 

second code is the pipeline for testing , implented directly on kaggle


our score and ranking in this competetion doesn't reflect really the score of light lgbm ( our final score is based on novomber submiision ) ,before moving to a transformrers 


Saint ++ is a transformers implemnted on pytorch based on Riid paper called SAINT , wich was an update of their previous work ( the main change in features used , positionel encoding , inputs in encoder block )

for training we have used vast ai "https://vast.ai/" ,renting a machine with four gpu , each trainng session take more than 7 hours 


 vivement les prochaine competetions 
 
