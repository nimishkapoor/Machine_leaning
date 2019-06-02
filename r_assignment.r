df1.df<-read.csv('train/user_data.csv')[,c('user_id','submission_count','problem_solved','max_rating','rating','rank')]
df2.df<-read.csv('train/problem_data.csv')[,c('problem_id','level_type')]
df3.df<-read.csv('train/train_submissions.csv')
df4.df<-read.csv('train/test_submissions.csv')

dfu<-df1.df[,]
dfu[, c(6)] <- sapply(dfu[, c(6)], as.numeric)

dfp<-df2.df[,]
dfp[, c(2)] <- sapply(dfp[, c(2)], as.numeric)

dftrain<-df3.df[,]

dftrain<-merge(x = dftrain, y = dfu, by = "user_id", all.x = TRUE)
dftrain<-merge(x = dftrain, y = dfp, by = "problem_id", all.x = TRUE)

dffinal<-df4.df[,]
dffinal<-merge(x = dffinal, y = dfu, by = "user_id", all.x = TRUE)
dffinal<-merge(x = dffinal, y = dfp, by = "problem_id", all.x = TRUE)

train=dftrain
train <- subset(train, select = c('submission_count','problem_solved','max_rating','rating','rank','level_type','attempts_range'))

fin=dffinal
fin<-subset(train, select = c('submission_count','problem_solved','max_rating','rating','rank','level_type'))

X=train

set.seed=1

y_train<-X[,7]
X_train<-X[,-7]

library(class)
k<-knn(X_train,fin,cl=y_train,k=18)

tb <- table(k,y_test)
accuracy <- function(x){sum(diag(x)/(sum(rowSums(x)))) * 100}
accuracy(tb)

summary(k)



