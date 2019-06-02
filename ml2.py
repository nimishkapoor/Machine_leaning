import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
import numpy as np

data=pd.read_csv('http://www-bcf.usc.edu/~gareth/ISL/Advertising.csv',index_col=0)
#print data.head()
#print data.tail()

#print data.shape

#sns.pairplot(data,x_vars=['TV','radio','newspaper'],y_vars='sales',size=7,aspect=0.7,kind='reg')
#plt.show()

#feature_cols=['TV','radio','newspaper']
#X=data[feature_cols]

#print type(X)
#print X.head()

y=data['sales']


#print type(y)
#print y.head()

#X_train, X_test, y_train, y_test = train_test_split(X,y,random_state=1)

#linreg=LinearRegression()
#linreg.fit(X_train,y_train)
#y_pred=linreg.predict(X_test)
 
#print linreg.intercept_
#print linreg.coef_

#print zip(feature_cols,linreg.coef_)

true=[100,50,30,20]
pred=[90,50,50,30]

#print (10+0+20+10)/4

#print metrics.mean_absolute_error(true,pred)
#print metrics.mean_squared_error(true,pred)

#print np.sqrt(metrics.mean_squared_error(true,pred))
#print np.sqrt(metrics.mean_squared_error(y_test,y_pred))



feature_cols=['TV','radio']
X=data[feature_cols]
X_train, X_test, y_train, y_test = train_test_split(X,y,random_state=1)
linreg=LinearRegression()
linreg.fit(X_train,y_train)
y_pred=linreg.predict(X_test)
print np.sqrt(metrics.mean_squared_error(y_test,y_pred))
