# -*- coding: utf-8 -*-
"""
Created on Sat Jul 13 10:17:10 2019

@author: Girish Chandra
"""

import pandas as pd
import numpy as np

cars_data=pd.read_csv("cars.csv",header=None)
cars_data.head()
cars_data.shape
cars_data.columns=['buying','maint','doors','persons','lug_boot','safety','classes'] ##assigning header to the dataset

cars_data.isnull().sum()

cars_df=pd.DataFrame.copy(cars_data)
colname=cars_df.columns[:]
colname

from sklearn import preprocessing
le=preprocessing.LabelEncoder()

for x in colname:
    cars_df[x]=le.fit_transform(cars_df[x])

cars_df.head()

cars_data.classes.value_counts()
#acc=0
#good=1
#unacc=2
#vgood=3

X=cars_df.values[:,:-1]
Y=cars_df.values[:,-1]

from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
scaler.fit(X)
X=scaler.transform(X)

from sklearn.model_selection import train_test_split
#Split the data into test and train
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.3, random_state=10)


##Running Decision Tree Model

#Predicting using the Decision_Tree_Classifier
from sklearn.tree import DecisionTreeClassifier
model_DecisionTree=DecisionTreeClassifier(random_state=10,min_samples_leaf=3,max_depth=10)

#fit the model on the data and predict the values
model_DecisionTree.fit(X_train,Y_train)

Y_pred=model_DecisionTree.predict(X_test)
#print(Y_pred)
#print(list(zip(Y_test,Y_pred)))

from sklearn.metrics import confusion_matrix,accuracy_score,classification_report
print(confusion_matrix(Y_test,Y_pred))
print(accuracy_score(Y_test,Y_pred))
print(classification_report(Y_test,Y_pred))

print(list(zip(colname,model_DecisionTree.feature_importances_))) #After training the model##Variable importance check, higher values are better   

#generate the file (saved as text in directory) and upload the code(in text file) in webgraphviz.com to plot the decision tree
from sklearn import tree
with open("model_DecisionTree.txt", "w") as f:
    f = tree.export_graphviz(model_DecisionTree, feature_names=colname[:-1],out_file=f)


##using SVM
from sklearn import svm
svc_model=svm.SVC(kernel='rbf',C=1.0,gamma=0.1)
#from sklearn.linear_model import LogsiticRegression
#svc_model=LogisticRegression()
svc_model.fit(X_train,Y_train)
Y_pred=svc_model.predict(X_test)
print(list(Y_pred))

Y_pred_col=list(Y_pred)

from sklearn.metrics import confusion_matrix,accuracy_score,classification_report
print(confusion_matrix(Y_test,Y_pred))
print(accuracy_score(Y_test,Y_pred))
print(classification_report(Y_test,Y_pred))

##using logistic regression

from sklearn.linear_model import LogisticRegression
#create a model
classifier=LogisticRegression()
#fitting training data to the model
classifier.fit(X_train,Y_train)

Y_pred=classifier.predict(X_test)
#print(list(zip(Y_test,Y_pred)))

from sklearn.metrics import confusion_matrix,accuracy_score,classification_report
print(confusion_matrix(Y_test,Y_pred))
print(accuracy_score(Y_test,Y_pred))
print(classification_report(Y_test,Y_pred))


#predicting using the Bagging_Classifier
from sklearn.ensemble import ExtraTreesClassifier

model=(ExtraTreesClassifier(50,random_state=10))
#fit the model on the data and predict the values
model=model.fit(X_train,Y_train)

Y_pred=model.predict(X_test)

from sklearn.metrics import confusion_matrix,accuracy_score,classification_report
print(confusion_matrix(Y_test,Y_pred))
print(accuracy_score(Y_test,Y_pred))
print(classification_report(Y_test,Y_pred))

##Randomforest
from sklearn.ensemble import RandomForestClassifier

model_RandomForest=(RandomForestClassifier(100,random_state=10))
#fit the model on the data and predict the values
model_RandomForest.fit(X_train,Y_train)

Y_pred=model_RandomForest.predict(X_test)

from sklearn.metrics import confusion_matrix,accuracy_score,classification_report
print(confusion_matrix(Y_test,Y_pred))
print(accuracy_score(Y_test,Y_pred))
print(classification_report(Y_test,Y_pred))




































































































