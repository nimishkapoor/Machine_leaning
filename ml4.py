import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn import metrics

data=pd.read_csv('http://www-bcf.usc.edu/~gareth/ISL/Advertising.csv',index_col=0)

feature_cols=['TV','radio','newspaper']
X=data[feature_cols]
y=data.sales

lm=LinearRegression()
scores=cross_val_score(lm,X,y,cv=10, scoring='neg_mean_squared_error')
#print scores
mse_scores=-scores
#print mse_scores

rmse_score=np.sqrt(mse_scores)
print rmse_score

print rmse_score.mean()

feature_cols1=['TV','radio']
X1=data[feature_cols1]
y1=data.sales
print np.sqrt(-cross_val_score(lm,X1,y1,cv=10,scoring='neg_mean_squared_error')).mean()