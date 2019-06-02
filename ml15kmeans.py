from sklearn.cluster import KMeans
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

df=pd.read_csv('https://raw.githubusercontent.com/codebasics/py/master/ML/13_kmeans/income.csv')

print df.head()

plt.scatter(df.Age,df['Income($)'])

plt.show()

km=KMeans(n_clusters=3)
y_pred=km.fit_predict(df.drop(['Name'],axis='columns'))

print y_pred

y_pred=km.fit_predict(df[['Age','Income($)']])
print y_pred

df['cluster']=y_pred
print df.head()

df1=df[df.cluster==0]
df2=df[df.cluster==1]
df3=df[df.cluster==2]
plt.scatter(df1.Age,df1['Income($)'],color='green')
plt.scatter(df2.Age,df2['Income($)'],color='blue')
plt.scatter(df3.Age,df3['Income($)'],color='black')
plt.xlabel('Age')
plt.ylabel('Income($)')
plt.show()

scalar=MinMaxScaler()
scalar.fit(df[['Income($)']])
df['Income($)']=scalar.transform(df[['Income($)']])
print df.head()

scalar.fit(df[['Age']])
df.Age=scalar.transform(df[['Age']])
print df.head()
plt.scatter(km.cluster_centers_[:,0])
km=KMeans(n_clusters=3)
y_pred=km.fit_predict(df.drop(['Name'],axis='columns'))
y_pred=km.fit_predict(df[['Age','Income($)']])
print y_pred

df['cluster']=y_pred
df1=df[df.cluster==0]
df2=df[df.cluster==1]
df3=df[df.cluster==2]
plt.scatter(df1.Age,df1['Income($)'],color='green')
plt.scatter(df2.Age,df2['Income($)'],color='blue')
plt.scatter(df3.Age,df3['Income($)'],color='black')
plt.xlabel('Age')
plt.ylabel('Income($)')
plt.scatter(km.cluster_centers_[:,0],km.cluster_centers_[:,1],color='purple',marker='*',label='centroid')

plt.show()

k_rng=range(1,10)
sse=[]
for k in k_rng:
	km=KMeans(n_clusters=k)
	km.fit(df[['Age','Income($)']])
	sse.append(km.inertia_)

print sse

plt.xlabel('K')
plt.ylabel('SSE')
plt.plot(k_rng,sse)
plt.show()	