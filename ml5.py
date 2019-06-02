from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_iris
from sklearn import metrics
import matplotlib.pyplot as plt
import pandas as pd
k_range = range(1, 31)

#param_grid=dict(n_neighbors=k_range)
#print param_grid

knn=KNeighborsClassifier()
#grid=GridSearchCV(knn,param_grid,cv=10,scoring='accuracy')

iris=load_iris()
X=iris.data
y=iris.target

#grid.fit(X,y)
#print grid.best_score_
#print grid.best_params_
#print grid.best_estimator_

#print pd.DataFrame(grid.cv_results_)[['mean_test_score','std_test_score','params']]
#print grid.cv_results_['params'][0]
#print grid.cv_results_['mean_test_score'][0]

#grid_mean_scores=grid.cv_results_['mean_test_score']
#print grid_mean_scores

#plt.plot(k_range,grid_mean_scores)
#plt.xlabel('Values of K')
#plt.ylabel('Cross Validation score')
#plt.show()

weight_options=['uniform','distance']

#param_grid=dict(n_neighbors=k_range,weights=weight_options)
#print param_grid

#grid=GridSearchCV(knn,param_grid,cv=10,scoring='accuracy')
#grid.fit(X,y)

#print grid.cv_results_
#print grid.best_score_
#print grid.best_params_

#print  grid.predict([[3,5,4,2]])
param_dist=dict(n_neighbors=k_range,weights=weight_options)
rand = RandomizedSearchCV(knn,param_dist,cv=10,scoring='accuracy',n_iter=10,random_state=1)
rand.fit(X,y)
print rand.best_score_
print rand.best_params_

