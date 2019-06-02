import pandas as pd
from sklearn.datasets import load_digits
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
import seaborn as sn

digits=load_digits()

print dir(digits)

plt.gray()
for i in range(4):
	plt.matshow(digits.images[i])
	#plt.show()

df=pd.DataFrame(digits.data)
#print df.head()
print digits.target
df['target']=digits.target

#print df.head()
X=df.drop(['target'],axis='columns')
y=digits.target

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2)

rf=RandomForestClassifier(n_estimators=30)
rf.fit(X_train,y_train)
y_pred=rf.predict(X_test)

print rf.score(X_test,y_test)
print metrics.accuracy_score(y_test,y_pred)

cm=metrics.confusion_matrix(y_test,y_pred)

plt.figure(figsize=(10,7))
sn.heatmap(cm,annot=True)
plt.xlabel('Predicted')
plt.ylabel('Truth')
plt.show()
