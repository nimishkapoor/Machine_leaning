from dateutil.parser import parse
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
plt.rcParams.update({'figure.figsize':(10,7),'figure.dpi':120})

df=pd.read_csv('https://raw.githubusercontent.com/selva86/datasets/master/a10.csv',parse_dates=['date'],index_col='date')

print df.head()

def plot_df(df,x,y,title="",xlabel='Date',ylabel='Value',dpi=100):
	plt.figure(figsize=(16,5),dpi=dpi)
	plt.plot(x,y,color='tab:red')
	plt.gca().set(title=title,xlabel=xlabel,ylabel=ylabel)
	plt.show()

plot_df(df,x=df.index,y=df.value,title='Monthly anti-diabetic drug sales in Australia from 1992 to 2008.')


df.reset_index(inplace=True)

df['year']=[d.year for d in df.date]
df['month']=[d.month for d in df.date]
years=df['year'].unique()

np.random.seed(100)
mycolors=np.random.choice(list(mpl.colors.XKCD_COLORS.keus()),len(years),replace=False)

plt.figure(figsize=(16,12),dpi=80)
for i,y in enumerate(years):
	if i>0:
		plt.plot('month','value',data.df.loc[df.year])