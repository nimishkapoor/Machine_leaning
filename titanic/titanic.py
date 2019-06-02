srst=from sklearn import metrics
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import Perceptron
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt


train=pd.read_csv('train.csv',index_col='PassengerId')
test=pd.read_csv('test.csv')
tmp=test
target=train['Survived']
train=train.drop(['Name','Survived','Ticket','Cabin'],axis='columns')

train['Sex']=train['Sex'].replace({'male':0,'female':1})
train['Embarked']=train['Embarked'].replace({'S':1,'Q':2,'C':3})

train.fillna(train.mean(),inplace=True)

test=test.drop(['Name','Ticket','Cabin','PassengerId'],axis='columns')

test['Sex']=test['Sex'].replace({'male':0,'female':1})
test['Embarked']=test['Embarked'].replace({'S':1,'Q':2,'C':3})

test.fillna(test.mean(),inplace=True)

models=[]
models.append(('CART',DecisionTreeClassifier()))
models.append(('RF',RandomForestClassifier()))
models.append(('LR',LogisticRegression()))
models.append(('PPN',Perceptron()))
models.append(('NB',GaussianNB()))
models.append(('SVM',SVC()))

results=[]
names=[]

for name,model in models:
	scores=cross_val_score(model,train,target,cv=10,scoring='accuracy')
	results.append(scores.mean())
	names.append(name)

print names
print results

#fig=plt.figure()
#fig.suptitle('Algorithm Comparison')
#ax=fig.add_subplot(111)
#plt.boxplot(results)
#ax.set_xticklabels(names)
#plt.show()

#plt.plot(names,results)
#plt.xlabel('Model')
#plt.ylabel('Testing Accuracy Score')
#plt.show()

RF=RandomForestClassifier()
RF.fit(train,target)
result=RF.predict(test)

print result

sub=pd.DataFrame(tmp['PassengerId'])
sub['Survived']=result
sub.columns=['PassengerId','Survived']
print sub.head()
print sub.shape

sub.to_csv('upload.csv',encoding='utf-8',index=False)
