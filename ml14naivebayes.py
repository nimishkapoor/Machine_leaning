import numpy as np
import pandas as pd
import urllib

from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB

from sklearn.model_selection import train_test_split
from sklearn import metrics

raw_data=urllib.urlopen('http://archive.ics.uci.edu/ml/machine-learning-databases/spambase/spambase.data')
df=np.loadtxt(raw_data,delimiter=',')
print df[0]

X=df[:,0:48]
y=df[:,-1]

X_train, X_test, y_train, y_test=train_test_split(X,y,test_size=.33,random_state=17)

BernNB=BernoulliNB(binarize=0.1)
BernNB.fit(X_train,y_train)
y_pred=BernNB.predict(X_test)
print metrics.accuracy_score(y_test,y_pred)

GauNB=GaussianNB()
GauNB.fit(X_train,y_train)
y_pred=GauNB.predict(X_test)
print metrics.accuracy_score(y_test,y_pred)

MultiNB=MultinomialNB()
MultiNB.fit(X_train,y_train)
y_pred=MultiNB.predict(X_test)
print metrics.accuracy_score(y_test,y_pred)