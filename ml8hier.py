import numpy as np
import pandas as pd
import scipy
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.cluster.hierarchy import fcluster
from scipy.cluster.hierarchy import cophenet
from scipy.spatial.distance import pdist

import seaborn as sb
import matplotlib.pyplot as plt

from sklearn.cluster import AgglomerativeClustering
from sklearn import metrics

np.set_printoptions(precision=4,suppress=True)
plt.figure(figsize=(10,3))
plt.style.use('seaborn-whitegrid')


address='data/mtcars.csv'
cars=pd.read_csv(address)
cars.columns=['car_names','mpg','cyl','disp','hp','drat','wt','qsec','vs','am','gear','carb']
X=cars.ix[:,(1,3,4,6)].values
y=cars.ix[:,(9)].values

print X
print y

Z=linkage(X,'ward')
dendrogram(Z,truncate_mode='lastp',p=12,leaf_rotation=45.,leaf_font_size=1)
plt.title('dendrogram')
plt.xlabel('Cluster Size')
plt.ylabel('Distance')

plt.axhline(y=500)
plt.axhline(y=50)

#plt.show()

k=2
HCluster= AgglomerativeClustering(n_clusters=k,affinity='euclidean',linkage='ward')
HCluster.fit(X)

print metrics.accuracy_score(y,HCluster.labels_)

HCluster= AgglomerativeClustering(n_clusters=k,affinity='euclidean',linkage='complete')
HCluster.fit(X)

print metrics.accuracy_score(y,HCluster.labels_)

HCluster= AgglomerativeClustering(n_clusters=k,affinity='euclidean',linkage='average')
HCluster.fit(X)

print metrics.accuracy_score(y,HCluster.labels_)

HCluster= AgglomerativeClustering(n_clusters=k,affinity='manhattan',linkage='average')
HCluster.fit(X)

print metrics.accuracy_score(y,HCluster.labels_)