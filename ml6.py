import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.preprocessing import binarize
from sklearn.model_selection import cross_val_score

path='data/pima-indians-diabetes.data'
col_names=['pregnent','glucose','bp','skin','insulin','bmi','peigree','age','label']
pima=pd.read_csv(path,header=None,names=col_names)
print pima.head()

feature_cols=['pregnent','insulin','bmi','age']

X=pima[feature_cols]
y=pima.label

X_train, X_test, y_train, y_test = train_test_split(X,y,random_state=0);

logreg=LogisticRegression()
logreg.fit(X_train,y_train)

y_pred=logreg.predict(X_test)

print metrics.accuracy_score(y_test,y_pred)

print y_test.value_counts()
print y_test.mean()
print 1-y_test.mean()
print max(y_test.mean(),1-y_test.mean())

##for multiclass classification null accuracy
print y_test.value_counts().head(1)/len(y_test)

print "True", y_test.values[0:25]
print "Pred", y_pred [0:25]

confusion=metrics.confusion_matrix(y_test,y_pred)
print confusion
tp=confusion[1,1]
tn=confusion[0,0]
fp=confusion[0,1]
fn=confusion[1,0]

print tp
print tn
print fp
print fn

print (tp+tn)/float(tp+fp+tn+fn)
print metrics.accuracy_score(y_test,y_pred)

print (fp+fn)/float(tp+tn+fp+fn)
print 1- metrics.accuracy_score(y_test,y_pred)

print tp/float (tp+fn)
print metrics.recall_score(y_test,y_pred)

print tn/float(tn+fp)

print metrics.precision_score(y_test,y_pred)

print logreg.predict(X_test)[0:10]
print logreg.predict_proba(X_test)[0:10,:]
print logreg.predict_proba(X_test)[0:10,1]
y_pred_prob= logreg.predict_proba(X_test)[:,1]

plt.rcParams['font.size']=14

plt.hist(y_pred_prob,bins=8)
plt.xlim(0,1)
plt.title("Histogram predicted probabilities.")
plt.xlabel('Predicted probabilities of diabetes')
plt.ylabel('Frequency')
#plt.show()

y_pred=binarize([y_pred_prob],0.3)[0]
print y_pred_prob[0:10]

print y_pred[0:10]

print confusion
print metrics.confusion_matrix(y_test,y_pred)

fpr, tpr, thresholds =metrics.roc_curve(y_test,y_pred_prob)
plt.plot(fpr,tpr)
plt.xlim(0.0,1.0)
plt.ylim(0.0,1.0)
plt.title('ROC curve for diabetes classifier!')
plt.xlabel('False Positive rate (1-Specificity)')
plt.ylabel('True Positive rate(Senstivity)')
plt.grid(True)
#plt.show()

def evaluate_threshold(threshold):
	print 'Senstivity:', tpr[thresholds>threshold][-1]
	print 'Specificity', 1-fpr[thresholds>threshold][-1]
evaluate_threshold(0.5)
evaluate_threshold(0.3)

print metrics.roc_auc_score(y_test,y_pred_prob)

print cross_val_score(logreg,X,y,cv=10,scoring='roc_auc').mean()