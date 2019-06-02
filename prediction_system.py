'import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

df1=pd.read_csv('train/user_data.csv')
df2=pd.read_csv('train/problem_data.csv')
df3=pd.read_csv('train/train_submissions.csv')
df4=pd.read_csv('train/test_submissions.csv')
print df4.shape

dfu=df1.drop(['contribution','country','follower_count','last_online_time_seconds','registration_time_seconds','rank'],axis='columns')
dfu['rank']=df1['rank'].replace({'expert':4,'advanced':3,'intermediate':2,'beginner':1})

dfp=df2.drop(['points','tags','level_type'],axis='columns')
dfp['level_type']=df2['level_type'].replace({'A':1,'B':2,'C':3,'D':4,'E':5,'F':6,'G':7,'H':8,'I':9,'J':10,'K':11,'L':12,'M':13,'N':14})

dftrain=df3
dftrain=pd.merge(dftrain,dfu,on='user_id',how='left')
dftrain=pd.merge(dftrain,dfp,on='problem_id',how='left')


dffinal=df4
dffinal=pd.merge(dffinal,dfu,on='user_id',how='left')
dffinal=pd.merge(dffinal,dfp,on='problem_id',how='left')
print dffinal.shape

X=dftrain.drop(['user_id','problem_id','attempts_range'],axis='columns')
X.fillna(X.mean(), inplace=True)

y=dftrain['attempts_range']
y.fillna(y.mean(), inplace=True)
y = y.astype(int)

unique, counts = np.unique(y, return_counts=True)
print dict(zip(unique, counts))

X_final=dffinal.drop(['user_id','problem_id','ID'],axis='columns')
X_final.fillna(X_final.mean(), inplace=True)

#print X_train.head()
#print y_train.head()

seed=7

models=[]
models.append(('KNN',KNeighborsClassifier()))
models.append(('CART',DecisionTreeClassifier()))
models.append(('RF',RandomForestClassifier()))

results=[]
names=[]
for name,model in models:
	X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=seed)
	model.fit(X_train,y_train)
	y_pred=model.predict(X_test)
	res=metrics.accuracy_score(y_test,y_pred)
	results.append(res)
	names.append(name)

print names
print results

#fig=plt.figure()
#fig.suptitle('Algorithm Comparison')
#ax=fig.add_subplot(111)
#plt.boxplot(results)
#ax.set_xticklabels(names)
#plt.show()

#X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=seed)

#scores=[]
#for i in range (1,20):
#knn=KNeighborsClassifier(n_neighbors=18)
#knn.fit(X_train,y_train)
#y_pred=knn.predict(X_test)
#x= metrics.accuracy_score(y_test,y_pred)
#	scores.append(x)
#print scores

CART=KNeighborsClassifier(n_neighbors=18)
CART.fit(X_train,y_train)
result=CART.predict(X_final)

plt.plot(names,results)
plt.xlabel('Model')
plt.ylabel('Testing Accuracy Score')
plt.show()


#result=knn.predict(X_final)

print result
unique, counts = np.unique(y_train, return_counts=True)
print dict(zip(unique, counts))
unique, counts = np.unique(result, return_counts=True)
print dict(zip(unique, counts))

sub=pd.DataFrame(result)
sub['ID']=df4['ID']
sub.columns=['attempts_range','ID']
print sub.head()
print sub.shape
sub.to_csv('upload.csv',encoding='utf-8', index=False)
