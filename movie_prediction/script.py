import warnings
warnings.filterwarnings('ignore')

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from scipy import stats
from scipy import stats,special
from scipy import io

import time
import json
import ast

from sklearn import model_selection, metrics, linear_model, datasets, feature_selection
from sklearn import neighbors

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score, KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_validate
from sklearn.preprocessing import LabelEncoder
#import lightgbm as lgb
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


train=pd.read_csv('train.csv')
test=pd.read_csv('test.csv')

#plt.figure(figsize=(8,6))
#plt.scatter((train['budget']), train['revenue'])
#plt.title('Revenue vs Budget')
#plt.xlabel('Budget')
#plt.ylabel('Revenue')
#plt.show()

#plt.figure(figsize=(8,6))
#plt.scatter(np.log10(train['budget']), np.log10(train['revenue']))
#plt.title('Revenue vs Budget')
#plt.xlabel('Budget [log10]')
#plt.ylabel('Revenue [log10]')
#plt.show()

#plt.figure(figsize=(8,6))
#plt.scatter(np.log10(train['popularity']), np.log10(train['revenue']))
#plt.title('Revenue vs Popularity')
#plt.xlabel('Popularity [log10]')
#plt.ylabel('Revenue [log10]')
#plt.show()

#Comparing the movies with the biggest budget values:

#train.sort_values('budget',ascending=False).head(10).plot(x='original_title',y='budget',kind='barh')
#plt.xlabel('Budget')
#plt.show()

#Comparing the movies with the biggest revenue values:

#train.sort_values('revenue',ascending=False).head(10).plot(x='original_title',y='revenue',kind='barh')
#plt.xlabel('Revenue')
#plt.show()

#Comparing the movies with the biggest profit values:

#train.assign(profit=lambda df: df['revenue']-df['budget']).sort_values('profit',ascending=False).head(10).plot(x='original_title',y='profit',kind='barh')
#plt.xlabel('Profit')
#plt.show()

#Moving ahead to explore the highest Revenue by 'genres' as follow:
#train.groupby('genres')['revenue'].mean().sort_values().head(10).plot(kind='barh')
#plt.xlabel('Revenue');
#plt.show()

#train['collection']=~train['belongs_to_collection'].isna()
#print train['collection'].value_counts()

#fig, ax=plt.subplots(figsize=(8,6)) 
#ax.set_yscale('symlog')
#sns.boxplot(x= 'collection', y='revenue', data=train, ax=ax);
#plt.show()

def parse_json(x):
	try: return json.loads(x.replace("'",'"'))[0]['name']
	except: return ''

#print train['production_companies'].head(10)

#train['production_companies']=train['production_companies'].apply(parse_json)

#print train['production_companies'].head(10)

#train.groupby('production_companies')['revenue'].mean().sort_values(ascending=False).head(20).plot(kind='barh')
#plt.xlabel('Revenue');
#plt.show()

#analyser=SentimentIntensityAnalyzer()

#train['overview']=train['overview'].fillna('')
#train['tagline']=train['tagline'].fillna('')

#analyser.polarity_scores(train['overview'].iloc[0])

#train['sentiment']=train['overview'].apply(lambda x:analyser.polarity_scores(x.lower())['compound'])

#train.groupby(pd.cut(train['sentiment'],6))['revenue'].mean().plot(kind='barh')
#plt.show()

#train['tag_sentiment']=train['tagline'].apply(lambda x:analyser.polarity_scores(x.lower())['compound'])
#train.groupby(pd.cut(train['tagline_sentiment'],6))['revenue'].mean().plot(kind='barh')
#plt.show()

#print train[['tag_sentiment','sentiment']].corrwith(train['revenue'])

#Helper function to parse text and convert given strings to lists

def text_to_list(x):
	if pd.isna(x):
		return ''
	else:
		return ast.literal_eval(x)

ntrain=train.shape[0]
ntest=test.shape[0]

combined = pd.concat((train,test),sort=False)

combined.drop(columns=['id','imdb_id','poster_path','title','original_title'],inplace=True)

for col in ['genres','production_companies','production_countries','spoken_languages','Keywords','cast','crew']:
	combined[col]=combined[col].apply(text_to_list)

print combined['belongs_to_collection'].head()
combined['belongs_to_collection']=1*(~combined['belongs_to_collection'].isna())
print combined['belongs_to_collection'].head()

combined['tagline']=1*(~combined['tagline'].isna())
combined['homepage']=1*(~combined['homepage'].isna())

combined['overview']=combined['overview'].str.len()

combined['overview'].fillna(0,inplace=True)

print combined['overview'].head()

combined['genre_number']=combined['genres'].apply(lambda x:len(x))

print combined['genre_number'].head()

print combined['genre_number'].value_counts()


def parse_genre(x):
    if type(x) == str:
        return pd.Series(['','',''], index=['genres1', 'genres2', 'genres3'] )
    if len(x) == 1:
        return pd.Series([x[0]['name'],'',''], index=['genres1', 'genres2', 'genres3'] )
    if len(x) == 2:
        return pd.Series([x[0]['name'],x[1]['name'],''], index=['genres1', 'genres2', 'genres3'] )
    if len(x) > 2:
        return pd.Series([x[0]['name'],x[1]['name'],x[2]['name']], index=['genres1', 'genres2', 'genres3'] )

combined[['genres1', 'genres2', 'genres3']] = combined['genres'].apply(parse_genre)
combined.drop(columns='genres', inplace=True)

print combined['genres1'].head()

print combined['production_companies'].head()

combined['production_company_number']=combined['production_companies'].apply(lambda x:len(x))

print combined['production_company_number'].head()

def parse_production_companies(x):
    if type(x) == str:
        return pd.Series(['','',''], index=['prod1', 'prod2', 'prod3'] )
    if len(x) == 1:
        return pd.Series([x[0]['name'],'',''], index=['prod1', 'prod2', 'prod3'] )
    if len(x) == 2:
        return pd.Series([x[0]['name'],x[1]['name'],''], index=['prod1', 'prod2', 'prod3'] )
    if len(x) > 2:
        return pd.Series([x[0]['name'],x[1]['name'],x[2]['name']], index=['prod1', 'prod2', 'prod3'] )

combined[['prod1', 'prod2', 'prod3']] = combined['production_companies'].apply(parse_production_companies)
combined.drop(columns='production_companies', inplace=True)


combined['production_country_number']=combined['production_countries'].apply(lambda x:len(x))

def parse_production_countries(x):
	if type(x)==str:
		return pd.Series(['','',''], index=['country1','country2','country3'])
	if len(x)==1:
		return pd.Series([x[0]['name'],'',''],index=['country1','country2','country3'])
	if len(x)==2:
		return pd.Series([x[0]['name'],x[1]['name'],''],index=['country1','country2','country3'])
	if len(x)>2:
		return pd.Series([x[0]['name'],x[1]['name'],x[2]['name']],index=['country1','country2','country3'])	

combined[['country1','country2','country3']]=combined['production_countries'].apply(parse_production_countries)

combined.drop(columns='production_countries',inplace=True)


combined['release_date']=pd.to_datetime(combined['release_date'],format='%m/%d/%y')

combined['weekday']=combined['release_date'].dt.weekday
combined['weekday'].fillna(4,inplace=True)

combined['month']=combined['release_date'].dt.month
combined['month'].fillna(9,inplace=True)

combined['year']=combined['release_date'].dt.year
combined['year'].fillna(combined['year'].median(),inplace=True)

combined['day']=combined['release_date'].dt.day
combined['day'].fillna(1,inplace=True)

combined.drop(columns=['release_date'],inplace=True)

combined['runtime'].fillna(combined['runtime'].median(),inplace=True)

combined['spoken_languages_number']=combined['spoken_languages'].apply(lambda x:len(x))

def parse_spoken_languages(x):
    if type(x) == str:
        return pd.Series(['','',''], index=['lang1', 'lang2', 'lang3'])
    if len(x) == 1:
        return pd.Series([x[0]['name'],'',''], index=['lang1', 'lang2', 'lang3'])
    if len(x) == 2:
        return pd.Series([x[0]['name'],x[1]['name'],''], index=['lang1', 'lang2', 'lang3'])
    if len(x) > 2:
        return pd.Series([x[0]['name'],x[1]['name'],x[2]['name']], index=['lang1', 'lang2', 'lang3'])

combined[['lang1','lang2','lang3']]=combined['spoken_languages'].apply(parse_spoken_languages)
combined.drop(columns='spoken_languages',inplace=True)

combined['status'].fillna('Released',inplace=True)

combined['keywords_number'] = combined['Keywords'].apply(lambda x: len(x))

def parse_keywords(x):
    if type(x) == str:
        return pd.Series(['','',''], index=['key1', 'key2', 'key3'])
    if len(x) == 1:
        return pd.Series([x[0]['name'],'',''], index=['key1', 'key2', 'key3'])
    if len(x) == 2:
        return pd.Series([x[0]['name'],x[1]['name'],''], index=['key1', 'key2', 'key3'])
    if len(x) > 2:
        return pd.Series([x[0]['name'],x[1]['name'],x[2]['name']], index=['key1', 'key2', 'key3'])

combined[['key1', 'key2', 'key3']] = combined['Keywords'].apply(parse_keywords)
combined.drop(columns='Keywords', inplace=True)


combined['gender_0_number']=combined['cast'].apply(lambda row:sum([x['gender']==0 for x in row]))
combined['gender_1_number']=combined['cast'].apply(lambda row:sum([x['gender']==1 for x in row]))
combined['gender_2_number']=combined['cast'].apply(lambda row:sum([x['gender']==2 for x in row]))

combined['cast_number'] = combined['cast'].apply(lambda x: len(x))
def parse_cast(x):
	myindex=['cast1','cast2','cast3','cast4','cast5']
	out = [-1]*5

	if type(x)!=str:
		for i in range(min([5,len(x)])):
			out[i]=x[i]['id']

	return pd.Series(out, index=myindex)

combined[['cast1', 'cast2', 'cast3', 'cast4', 'cast5']] = combined['cast'].apply(parse_cast)
combined.drop(columns='cast', inplace=True)



combined['crew_number'] = combined['crew'].apply(lambda x: len(x))

def parse_crew(x):
	myindex=['Director','Producer']
	out=[-1]*2
	if type(x) != str:
		for item in x:
			if item['job']=='Director':
				out[0]=item['id']
			elif item['job']=='Producer':
				out[1]==item['id']

	return pd.Series(out,index=myindex)

combined[['Director', 'Producer']] = combined['crew'].apply(parse_crew)
combined.drop(columns='crew', inplace=True)

combined['budget_log']=np.log10(combined['budget'])
combined['pop_log']=np.log1p(combined['popularity'])


cols=['genres1','genres2','genres3']
allitems=list(set(combined[cols].values.ravel().tolist()))
labeler=LabelEncoder()
labeler.fit(allitems)
combined[cols]=combined[cols].apply(lambda x:labeler.transform(x))

cols = ['prod1', 'prod2', 'prod3']
allitems = list(set(combined[cols].values.ravel().tolist()))
labeler = LabelEncoder()
labeler.fit(allitems)
combined[cols] = combined[cols].apply(lambda x: labeler.transform(x))

cols = ['country1', 'country2', 'country3']
allitems = list(set(combined[cols].values.ravel().tolist()))
labeler = LabelEncoder()
labeler.fit(allitems)
combined[cols] = combined[cols].apply(lambda x: labeler.transform(x))

cols = ['lang1', 'lang2', 'lang3']
allitems = list(set(combined[cols].values.ravel().tolist()))
labeler = LabelEncoder()
labeler.fit(allitems)
combined[cols] = combined[cols].apply(lambda x: labeler.transform(x))

cols = ['key1', 'key2', 'key3']
print combined[cols].values
print combined[cols].values.ravel()

allitems = list(set(combined[cols].values.ravel().tolist()))
labeler = LabelEncoder()
labeler.fit(allitems)
combined[cols] = combined[cols].apply(lambda x: labeler.transform(x))


combined_dummy = combined.copy()
cat_col = combined.select_dtypes('object').columns
combined_dummy[cat_col] = combined_dummy[cat_col].apply(lambda x: LabelEncoder().fit_transform(x))


train_data=combined_dummy.iloc[:ntrain]
test_data=combined_dummy.iloc[-ntest:]

X_train=train_data.drop(columns='revenue').values

#y_train=np.log1p(train_data['revenue']).values

y_train=train_data['revenue']

X_test=test_data.drop(columns='revenue').values

from sklearn.preprocessing import Imputer
imputer = Imputer()
X_train = imputer.fit_transform(X_train) 
X_test =imputer.fit_transform(X_test)

#y_train = imputer.fit_transform(y_train)

#kf=KFold(n_splits=5,shuffle=True,random_state=123)
lr=LinearRegression()
#y_pred=cross_val_score(lr,X_train,y_train,cv=kf)
#y_pred[y_pred<0]=0

#print y_pred
#print('RMSLE:{0:.2f}'.format(np.sqrt(mean_squared_error(y_train,y_pred))))

rf=RandomForestRegressor(max_depth=20,random_state=123,n_estimators=100)
#y_pred=cross_val_score(rf,X_train,y_train,cv=kf)
#y_pred[y_pred<0]=0

rf.fit(X_train,y_train)
result=rf.predict(X_test)

sub=pd.DataFrame(test['id'])
sub['revenue']=result
sub.columns=['id','revenue']

sub.to_csv('upload.csv',encoding='utf-8',index=False)

#print('RMSLE: {0:.2f}'.format(np.sqrt(mean_squared_error(y_train,y_pred))))

#rf.fit(X_train,y_train)
#imp=pd.Series(rf.feature_importances_,index=train_data.drop(columns='revenue').columns)
#imp.sort_values(ascending=False).plot(kind='barh',figsize=(8,10))
#plt.show()

#lgg_model=lgb.LGBMRegressor(num_leaves=20,max_depth=-1,learning_rate=0.01,metrics='rmse',n_estimators=1500,feature_fraction=0.4)
#y_pred=cross_val_score(lgb_model,X_train,y_train,cv=kf)

#rint('RMSLE:{0:.2f}'.format(np.sqrt(mean_squared_error(y_train,y_pred))))

#lgb_model.fit(X_train,y_train)
#mp=pd.Series(lgb_model.feature_importances_,index=train_data.drop(columns='revenue').columns)
#imp.sort_values(ascending=False).plot(kind='barh',figsize=(8,10))
#plt.show(0)
