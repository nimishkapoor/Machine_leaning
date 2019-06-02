from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
import matplotlib.pyplot as plt

iris=load_iris()

X=iris.data
y=iris.target

X_train, X_test, y_train, y_test=train_test_split(X,y,random_state=1)

knn=KNeighborsClassifier(n_neighbors=20)
logreg=LogisticRegression()
#knn.fit(X_train,y_train)
#y_pred=knn.predict(X_test)
#print metrics.accuracy_score(y_test,y_pred)


#kf=KFold(25,n_folds=5,shuffle=False)

#scores=cross_val_score(knn,X,y,cv=10,scoring='accuracy')
#print scores
#print scores.mean()

#k_range = range(1,31)

#k_scores=[]
#for k in k_range:
#	knn=KNeighborsClassifier(n_neighbors=k)
#	scores=cross_val_score(knn,X,y,cv=10,scoring='accuracy')
#	k_scores.append(scores.mean())
#print k_scores
	
#plt.plot(k_range,k_scores)
#plt.xlabel('Value of K for KNN')
#plt.ylabel('Cross Validation Accuracy')
#plt.show()

print cross_val_score(knn,X,y,cv=10,scoring='accuracy').mean()
print cross_val_score(logreg,X,y,cv=10,scoring='accuracy').mean()

