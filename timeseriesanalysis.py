import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#from matplotlib.pylab import rcParams
#rcParams['figure.figsize']=10,6

df=pd.read_csv('https://datahack-prod.s3.amazonaws.com/train_file/Train_SU63ISt.csv')
print df.head()

indexeddf=df.set_index(['Datetime'])

from datetime import datetime
print indexeddf.head()

X=df['Datetime']
y=df['Count']

plt.xlabel('time')
plt.ylabel('count')

plt.plot(X,y)
#plt.show()

