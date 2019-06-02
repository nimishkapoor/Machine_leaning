from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


iris=load_iris()
#type(iris)
#print iris.data
#print iris.feature_names
#print iris.target
#print type(iris.data)
#print type(iris.target)
#print iris.data.shape
#print iris.target.shape

X=iris.data
y=iris.target

#print X.shape
#print y.shape

#knn = KNeighborsClassifier(n_neighbors=1)
#knn = KNeighborsClassifier(n_neighbors=5)

#knn.fit(X,y)

#knn.predict([[3,5,4,2]])

#X_new=[[3,5,4,2],[5,4,3,2]]
#print(knn.predict(X_new))

logreg = LogisticRegression()
#logreg.fit(X,y)
#print(logreg.predict(X_new))

#y_pred=logreg.predict(X)

#print(y_pred)

#print(len(y_pred))

#y_pred=knn.predict(X)

#print metrics.accuracy_score(y,y_pred)


X_train, X_test, y_train, y_test= train_test_split(X,y,test_size=0.4,random_state=4)
#print X_train.shape
#print X_test.shape

#logreg.fit(X_train,y_train)
#y_pred=logreg.predict(X_test)

#knn.fit(X_train,y_train)
#y_pred=knn.predict(X_test)

#print metrics.accuracy_score(y_test,y_pred)

scores=[]
for k in range(1,26):
	knn=KNeighborsClassifier(n_neighbors=k)
	knn.fit(X_train,y_train)
	y_pred=knn.predict(X_test)
	scores.append(metrics.accuracy_score(y_test,y_pred))

plt.plot(range(1,26),scores)
plt.xlabel('Value of K for KNN')
plt.ylabel('Testing Accuracy')
plt.show()

knn=KNeighborsClassifier(n_neighbors=11)
knn.fit(X,y)
print(knn.predict([[3,5,4,2]]))