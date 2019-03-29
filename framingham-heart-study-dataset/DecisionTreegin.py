# -*- coding: utf-8 -*-
"""
Created on Thu Feb 21 13:53:26 2019

@author: Anvi Puri
"""

#Importing the libraries
import numpy as np
import pandas as pd
#import matplotlib.pyplot as plt
#import statsmodels.api as sm #For the estimation of many statistical models as well as to conduct statistical tests 
import scipy.stats as ss#contains probability distributions as well as statistical functions
#import seaborn as sb #A python data visualization library based on matplotlib
import sklearn
#import matplotlib.mlab as mlab#Numerical python functions written for compatibility with MATLAB commands with the same name
   
#Importing the dataset
dataset=pd.read_csv("framingham.csv")
dataset.drop(['education'],axis=1,inplace=True)
dataset.head()
dataset.isnull().sum()

#Counting number of columns with missing values
count=0
for i in dataset.isnull().sum(axis=1):
    if i>0:
        count=count+1
print('Total number of rows with missing values is ', count)
#print('since it is only',round((count/len(dataset.index))*100), 'percent of the entire dataset the rows with missing values are excluded.')
#dataset.dropna(axis=0,inplace=True)
#dataset.isnull().sum()

#Creating matrix of independent features
X=dataset.iloc[:,:-1].values

#Creating dependent variable vector
Y=dataset.iloc[:,14].values

# Taking care of missing data
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = 'NaN', strategy = 'most_frequent', axis = 0) #Observation:Out of the 3 strategies, most_frequent worked the best
imputer = imputer.fit(X[:, 0:14])
X[:, 0:14] = imputer.transform(X[:, 0:14])

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.25, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

# Fitting Decision Tree to the Training set
from sklearn.tree import DecisionTreeClassifier
classifier=DecisionTreeClassifier(criterion='gini',random_state=0)
classifier.fit(X_train,y_train)

#Converting Input from user to Dataset and getting the output
sample={'col0':[1],'col1':[ 40],'col2': [1],'col3':[4],'col4':[0],'col5':[0],'col6':[1],'col7':[1],'col8':[395],'col9':[100],'col10':[70],'col11':[23],'col12':[80],'col13':[70]}
sample_re= pd.DataFrame(data=sample)
sample_re=sc_X.transform(sample_re)
y_re=classifier.predict(sample_re)
print(y_re)
print(type(y_re))

if(y_re==[0]):
    print ('hey')

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

# Applying k-Fold Cross Validation
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10)
print(accuracies.mean())
accuracies.std()
'''
# Applying Grid Search to find the best model and the best parameters
from sklearn.model_selection import GridSearchCV
parameters = [{'criterion': ['gini'], 'splitter': ['best','random'], 'max_depth':[10,100,500]},
              {'criterion': ['entropy'], 'splitter': ['best','random'],'max_depth':[10,100,500]}]
grid_search = GridSearchCV(estimator = classifier,
                         param_grid = parameters,
                         scoring = 'accuracy',
                       cv = 10,
                     n_jobs = -1)
grid_search = grid_search.fit(X_train, y_train)
best_accuracy = grid_search.best_score_
best_parameters = grid_search.best_params_
'''
#Accuracy
print(sklearn.metrics.accuracy_score(y_test,y_pred))


