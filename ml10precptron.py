import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score
from matplotlib.colors import ListedColormap
from matplotlib import rcParams
rcParams["figure.figsize"]=10,5

iris=datasets.load_iris()
X=iris.data
y=iris.target

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=0)

sc=StandardScaler()

sc.fit(X_train)

X_train_std=sc.transform(X_train)
X_test_std=sc.transform(X_test)

X_test_std=X_test_std[:,[2,3]]
X_train_std=X_train_std[:,[2,3]]

ppn=Perceptron(n_iter=40,eta0=0.1,random_state=0)
ppn.fit(X_train_std,y_train)

y_pred=ppn.predict(X_test_std)

print accuracy_score(y_test,y_pred)