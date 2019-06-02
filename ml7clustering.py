import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn import metrics
from sklearn.preprocessing import scale
from mpl_toolkits.mplot3d import Axes3D
from sklearn.datasets import load_iris

iris=load_iris()
X=scale(iris.data)
y=pd.DataFrame(iris.target)
variable_names = iris.feature_names
print X[0:10,]

cluster=KMeans(n_clusters=3,random_state=5)
cluster.fit(X)

iris_df=pd.DataFrame(iris.data)
iris_df.columns=['Sepal_Length','Sepal_Width','Petal_Length','Petal_Width']
y.columns=['Targets']

color_theme = np.array(['darkgray','lightsalmon','powderblue'])

plt.subplot(1,2,1)
plt.scatter(x=iris_df.Petal_Length,y=iris_df.Petal_Width,c=color_theme[iris.target],s=50)
#plt.show()

plt.subplot(1,2,2)
plt.scatter(x=iris_df.Petal_Length,y=iris_df.Petal_Width,c=color_theme[cluster.labels_],s=50)
#plt.show()

relabel=np.choose(cluster.labels_,[2,0,1]).astype(np.int64)
plt.subplot(1,2,1)
plt.scatter(x=iris_df.Petal_Length,y=iris_df.Petal_Width,c=color_theme[iris.target],s=50)
#plt.show()

plt.subplot(1,2,2)

plt.scatter(x=iris_df.Petal_Length,y=iris_df.Petal_Width,c=color_theme[cluster.labels_],s=50)
#plt.show()

print metrics.classification_report(y,relabel)


