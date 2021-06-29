# -*- coding: utf-8 -*-
"""
Created on Sat Nov 21 12:42:14 2020

@author: DEBJANI GHOSH
"""
#DECISION TREE CLASSIFIER.
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
df=pd.read_csv('D:\Titanic_data.csv')
new_df=df.drop(['PassengerId','Name','SibSp','Parch','Ticket','Cabin','Embarked'],axis='columns')
new_df.head()
print(new_df.isna().sum())
x= new_df.drop(['Survived'],axis='columns')
y=new_df['Survived']
x.Sex=x.Sex.map({'male':1,'female':2})
x.Age=x.Age.fillna(x.Age.mean())
print(x.isna().sum())

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=1)

#decision tree classifier
classifier=DecisionTreeClassifier()
classifier.fit(x_train,y_train)
y_pred=classifier.predict(x_test)
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_pred)
print(cm)
import matplotlib.pyplot as plt
import seaborn as sns
plt.figure(figsize=(10,7))
sns.heatmap(cm,annot=True)
plt.xlabel("predicted")
plt.ylabel("truth value")
print("accuracy of decision tree classifier is:",accuracy_score(y_test,y_pred))


#logistic regression
model=LogisticRegression()
model.fit(x_train,y_train)
pred=model.predict(x_test)
cm1=confusion_matrix(y_test,pred)
print(cm1)
plt.figure(figsize=(8,4))
sns.heatmap(cm1,annot=True)
plt.xlabel("predicted value")
plt.ylabel("truth value")
print("accuracy of logistic regression is:",accuracy_score(y_test,pred))
