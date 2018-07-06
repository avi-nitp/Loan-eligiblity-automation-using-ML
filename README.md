# Loan-eligiblity-automation-using-ML
Predicting loan eligibility process (real-time) of a loan provider company based on the customer details provided through an online application form.
#Idea
Since we need to classify weather a person loan application will be approved or not based on details provided,we use Logistic regression to predict the eligiblity
#Steps
First we preprocessed the dataset.Any missing value was replaced with mode(if data was categorical) and mean/median(if data was numeric).
Since it is a classification problem as we have to predict weather a person loan application will be accepted or not,we use logistic regression model for this purpose.
Then we changed any categorical data(eg Gender,Qualification)present into dummy variables using get_dummy() of pandas as logistic regression work on numbers.
We divided the data set into train and test and fit it to our model
An accuracy of 79% was achieved
