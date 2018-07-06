import sklearn
import pandas as pd
import numpy as np 
import seaborn as sns                  # For data visualization
import matplotlib.pyplot as plt        # For plotting graphs
%matplotlib inline
import warnings                        # To ignore any warnings
warnings.filterwarnings("ignore")

train=pd.read_csv(r"C:\Users\BRO\Desktop\train_u6lujuX_CVtuZ9i.csv")     #Reading Data
test=pd.read_csv(r"C:\Users\BRO\Desktop\test_Y3wMUE5_7gLdaTN.csv")

train_original=train.copy()            #keeping copy of original data 
test_original=test.copy()

'''
train['Loan_Status'].value_counts().plot.bar()                   #for data visualization
Gender=pd.crosstab(train['Gender'],train['Loan_Status'])
Gender.div(Gender.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True, figsize=(4,4))
'''
train.isnull().sum()    #now checking for any missing data

#for categorical data we fill the missing data with mode
train['Gender'].fillna(train['Gender'].mode()[0], inplace=True)
train['Married'].fillna(train['Married'].mode()[0], inplace=True)
train['Dependents'].fillna(train['Dependents'].mode()[0], inplace=True)
train['Self_Employed'].fillna(train['Self_Employed'].mode()[0], inplace=True)
train['Credit_History'].fillna(train['Credit_History'].mode()[0], inplace=True)
train['Loan_Amount_Term'].fillna(train['Loan_Amount_Term'].mode()[0], inplace=True)

#for numerical data we fill the missing data with mean or median of remaining values
train['LoanAmount'].fillna(train['LoanAmount'].median(), inplace=True)    #we use median here as mean is affected by outliers

#train.isnull().sum()

#we do the same with train dataset
test['Gender'].fillna(train['Gender'].mode()[0], inplace=True)
test['Dependents'].fillna(train['Dependents'].mode()[0], inplace=True)
test['Self_Employed'].fillna(train['Self_Employed'].mode()[0], inplace=True)
test['Credit_History'].fillna(train['Credit_History'].mode()[0], inplace=True)
test['Loan_Amount_Term'].fillna(train['Loan_Amount_Term'].mode()[0], inplace=True)
test['LoanAmount'].fillna(train['LoanAmount'].median(), inplace=True)

train['LoanAmount_log'] = np.log(train['LoanAmount'])  #remove any outliers
#train['LoanAmount_log'].hist(bins=20)
test['LoanAmount_log'] = np.log(test['LoanAmount'])

train=train.drop('Loan_ID',axis=1) #drop loan id as it does not affect output
test=test.drop('Loan_ID',axis=1)

X = train.drop('Loan_Status',1) #separating the data in X and y
y = train.Loan_Status

X=pd.get_dummies(X)      #converting categorical values in numerical form as logistic regression work on nos
train=pd.get_dummies(train)
test=pd.get_dummies(test)

from sklearn.model_selection import train_test_split
x_train, x_cv, y_train, y_cv = train_test_split(X,y, test_size =0.3)

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
model = LogisticRegression()  #model
model.fit(x_train, y_train)

LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
          intercept_scaling=1, max_iter=100, multi_class='ovr', n_jobs=1,
          penalty='l2', random_state=1, solver='liblinear', tol=0.0001,
          verbose=0, warm_start=False)
accuracy_score(y_cv,pred_cv)

pred_test = model.predict(test)

#submission part

#####################
result=pd.read_csv(r"C:\Users\BRO\Desktop\Sample_Submission_ZAuTl8O_FK3zQHh.csv")
result['Loan_Status']=pred_test
result['Loan_ID']=test_original['Loan_ID']

result['Loan_Status'].replace(0, 'N',inplace=True)
result['Loan_Status'].replace(1, 'Y',inplace=True)
pd.DataFrame(result, columns=['Loan_ID','Loan_Status']).to_csv('logistic.csv')
#####################
print(np.matrix(result))
