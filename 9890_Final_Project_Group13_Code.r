rm(list=ls())
cat("\014")
graphics.off()
library(ggplot2)
library(gridExtra)
library(glmnet)
library(caret)
library(readr)
library(tictoc)
library(MASS)
library(tree)
library(randomForest)
library(httr)

# Load the data
df<- read_csv("https://raw.githubusercontent.com/WZhang-05112021/STA9890_Final_Project_Group13/main/STA9890_Group13_Data.csv")
df<-as.data.frame(df[2:106])

set.seed(1)
n=dim(df)[1]
p=104

# 100 times sampling and CV for EN, Lasso and Ridge

X<-data.matrix(df[2:105])
y<-data.matrix(df$Score)

n.train        =     floor(0.8*n)
n.test         =     n-n.train

M              =     100
Rsq.test.ls    =     rep(0,M) 
Rsq.train.ls   =     rep(0,M)
Rsq.test.en    =     rep(0,M)  
Rsq.train.en   =     rep(0,M)
Rsq.test.rid   =     rep(0,M)
Rsq.train.rid  =     rep(0,M)
tic('100-sample')
for (m in c(1:M)) {
  shuffled_indexes =     sample(n)
  train            =     shuffled_indexes[1:n.train]
  test             =     shuffled_indexes[(1+n.train):n]
  X.train          =     X[train,]
  y.train          =     y[train]
  X.test           =     X[test,]
  y.test           =     y[test]
  a=0.5 # elastic-net 0<a<1
  cv.fiten         =     cv.glmnet(X.train, y.train, intercept = TRUE, alpha = a, nfolds = 10)
  fit              =     glmnet(X.train, y.train,intercept = TRUE, alpha = a, lambda = cv.fiten$lambda.min)
  y.train.hat      =     predict(fit, newx = X.train, type = "response") # y.train.hat=X.train %*% fit$beta + fit$a0
  y.test.hat       =     predict(fit, newx = X.test, type = "response") # y.test.hat=X.test %*% fit$beta  + fit$a0
  Rsq.test.en[m]   =     1-mean((y.test - y.test.hat)^2)/mean((y.test - mean(y.test))^2)
  Rsq.train.en[m]  =     1-mean((y.train - y.train.hat)^2)/mean((y.train - mean(y.train))^2)  
  lam.en           =   fit$lambda
  # fit lasso and calculate and record the train and test R squares 
  a=1 # lasso
  cv.fitls         =     cv.glmnet(X.train, y.train, intercept = TRUE, alpha = a, nfolds = 10)
  fit              =     glmnet(X.train, y.train,intercept = TRUE, alpha = a, lambda = cv.fitls$lambda.min)
  y.train.hat      =     predict(fit, newx = X.train, type = "response") # y.train.hat=X.train %*% fit$beta + fit$a0
  y.test.hat       =     predict(fit, newx = X.test, type = "response") # y.test.hat=X.test %*% fit$beta  + fit$a0
  Rsq.test.ls[m]   =     1-mean((y.test - y.test.hat)^2)/mean((y.test - mean(y.test))^2)
  Rsq.train.ls[m]  =     1-mean((y.train - y.train.hat)^2)/mean((y.train - mean(y.train))^2)   
  lam.las          =  fit$lambda
  a=0  # rid
  cv.fitrid        =     cv.glmnet(X.train, y.train, intercept = TRUE, alpha = a, nfolds = 10)
  fit              =     glmnet(X.train, y.train,intercept = TRUE, alpha = a, lambda = cv.fitrid$lambda.min)
  y.train.hat      =     predict(fit, newx = X.train, type = "response") # y.train.hat=X.train %*% fit$beta + fit$a0
  y.test.hat       =     predict(fit, newx = X.test, type = "response") # y.test.hat=X.test %*% fit$beta  + fit$a0
  Rsq.test.rid[m]   =     1-mean((y.test - y.test.hat)^2)/mean((y.test - mean(y.test))^2)
  Rsq.train.rid[m]  =     1-mean((y.train - y.train.hat)^2)/mean((y.train - mean(y.train))^2)
  lam.rid           =     fit$lambda
  
  cat(sprintf("m=%3.f| Rsq.test.ls=%.2f,  Rsq.test.en=%.2f, Rsq.test.rid=%.2f| Rsq.train.ls=%.2f,  Rsq.train.en=%.2f, Rsq.train.rid=%.2f| \n", m, Rsq.test.ls[m], Rsq.test.en[m],  Rsq.test.rid[m],Rsq.train.ls[m], Rsq.train.en[m],Rsq.train.rid[m]))
}

toc()

#4(c)
K = 10
d = ceiling(n/K)
# set.seed(0)
i.mix = sample(1:n)

#lasso

tic("10-fold")

for (k in 1:K) {
  cat("Fold",k,"\n")
  
  folds=(1+(k-1)*d):(k*d);
  i.tr=i.mix[-folds]
  i.val=i.mix[folds]
  
  X.tr = X[i.tr,]   
  y.tr = y[i.tr]    
  X.val = X[i.val,] 
  y.val = y[i.val]  
  
  cv.fit               =     cv.glmnet(X.tr, y.tr, intercept = FALSE, alpha = 1, nfolds = K)
  fit                  =     glmnet(X.tr, y.tr,intercept = FALSE, alpha = 1, lambda = cv.fit$lambda.min)
  y.train.hat.las      =     predict(fit, newx = X.tr, type = "response") # y.train.hat=X.train %*% fit$beta + fit$a0
  y.test.hat.las       =     predict(fit, newx = X.val, type = "response") 
}

toc()

# ridge
tic()

for (k in 1:K) {
  cat("Fold",k,"\n")
  
  folds=(1+(k-1)*d):(k*d);
  i.tr=i.mix[-folds]
  i.val=i.mix[folds]
  
  X.tr = X[i.tr,]   
  y.tr = y[i.tr]    
  X.val = X[i.val,] 
  y.val = y[i.val]  
  
  cv.fit           =     cv.glmnet(X.tr, y.tr, intercept = FALSE, alpha = 0, nfolds = 10)
  fit              =     glmnet(X.tr, y.tr,intercept = FALSE, alpha = 0, lambda = cv.fit$lambda.min)
  y.train.hat.rid      =     predict(fit, newx = X.tr, type = "response") 
  y.test.hat.rid       =     predict(fit, newx = X.val, type = "response") 
}

toc()

#elastic-net

tic()

for (k in 1:K) {
  cat("Fold",k,"\n")
  
  folds=(1+(k-1)*d):(k*d);
  i.tr=i.mix[-folds]
  i.val=i.mix[folds]
  
  X.tr = X[i.tr,]   
  y.tr = y[i.tr]    
  X.val = X[i.val,] 
  y.val = y[i.val]  
  
  cv.fit           =     cv.glmnet(X.tr, y.tr, intercept = FALSE, alpha = 0.5, nfolds = 10)
  fit              =     glmnet(X.tr, y.tr,intercept = FALSE, alpha = 0.5, lambda = cv.fit$lambda.min)
  y.train.hat.en      =     predict(fit, newx = X.tr, type = "response") 
  y.test.hat.en       =     predict(fit, newx = X.val, type = "response") 
}

toc()

# Random Forest: 100 times sampling and model fitting

n  =   nrow(df)
p=dim(df)[2]-1

M              =     100
Rsq.test.rf    =     rep(0,M)  # rf = random forest
Rsq.train.rf   =     rep(0,M)

for (m in c(1:M)) {
  
  train = sample(1:n, floor(0.8*n))
  
  rf.score    =  randomForest(Score ~ ., data=df, subset = train, mtry= floor(sqrt(p)), importance=TRUE)
  
  y.train     = df[train, "Score"]
  
  y.train.hat =  predict(rf.score, newdata = df[train,])
  
  y.test      = df[-train, "Score"]
  
  y.test.hat  = predict(rf.score, newdata = df[-train,])
  
  Rsq.test.rf[m]    =     1-    mean(   unlist(  (y.test - y.test.hat)^2)   )   /   mean(  unlist(  (    y.test - mean(unlist(y.test))    )^2  ) )
  Rsq.train.rf[m]   =     1-    mean(  unlist( (y.train - y.train.hat)^2)   )  /   mean( unlist( (y.train - mean(  unlist(y.train))  )^2)   )
  
  cat(sprintf("m=%3.f| Rsq.test.rf=%.2f,  Rsq.train.rf=%.2f| \n", m,  Rsq.test.rf[m],  Rsq.train.rf[m]))
}


# Q4(b) Show the side-by-side boxplots of R2 test;R2 train. We want to see two panels. One for training, and the other for testing.

# test r^2 box plot
boxplot(Rsq.test.ls, Rsq.test.rid, Rsq.test.en, Rsq.test.rf, main="Test R^2", ylab="Rsq", names = c('Lasso','Ridge', "Elastic Net", "Random Forest"), col = c('plum2', 'slategray1', 'tan3','palegreen'))

# train r^2 box plot
boxplot(Rsq.train.ls, Rsq.train.rid, Rsq.train.en, Rsq.train.rf, main="Train R^2", ylab="Rsq", names = c('Lasso','Ridge', "Elastic Net", "Random Forest"), col = c('plum4', 'slategray4', 'tan4','palegreen4'))


# Q4(c) For one on the 100 samples, create 10-fold CV curves for lasso, elastic-net  = 0:5,ridge. Record and present the time it takes to cross-validate ridge/lasso/elastic-net regression.
# MSE ane Lambda plot
plot(cv.fiten)
plot(cv.fitrid)
plot(cv.fitls)

#Q4(d) For one on the 100 samples, show the side-by-side boxplots of train and test residuals (1 slide).

# Lasso residual boxplot
r_las_train=(y.train.hat.las-y.tr)
r_las_test =(y.test.hat.las-y.val)

col<-list(te=r_las_test, tr=r_las_train)
r_las=as.data.frame(lapply(col,'length<-', max(sapply(col,length))))
boxplot(r_las, horizontal = TRUE)

# Ridge residual boxplot
r_rid_train=(y.train.hat.rid-y.tr)
r_rid_test =(y.test.hat.rid-y.val)

col<-list(te=r_rid_test, tr=r_rid_train)
r_rid=as.data.frame(lapply(col, 'length<-', max(sapply(col,length))))
boxplot(r_rid, horizontal = TRUE)

r_en_train=(y.train.hat.en-y.tr)
r_en_test =(y.test.hat.en-y.val)

# Elastic-Net residual boxplot
col<-list(te=r_en_test, tr=r_en_train)
r_en=as.data.frame(lapply(col, 'length<-', max(sapply(col,length))))
boxplot(r_en, horizontal = TRUE)

#Q5(2) 90% test R2 interval
# 90% of the Rsq for three different models
# rid
quantile(Rsq.test.rid,probs=c(.05,.95))
#lasso
quantile(Rsq.test.ls, probs=c(.05,.95))
#elastic-net
quantile(Rsq.test.en, probs=c(.05,.95))
#Random Forest
quantile(Rsq.test.rf, probs=c(.05,.95))

# Q5(2) Fit all four models to the whole data set

# fit elastic net to the whole data set
tic("elastic-timing")
a=0.5 # elastic-net
cv.fit           =     cv.glmnet(X, y, alpha = a, nfolds = 10)
fit.en           =     glmnet(X, y, alpha = a, lambda = cv.fit$lambda.min)
beta.en.bs       =     as.vector(fit.en$beta)
toc()

# fit lasso to the whole data set
tic('lasso-timing')
a=1 # lasso
cv.fit           =     cv.glmnet(X, y, alpha = a, nfolds = 10)
fit.ls           =     glmnet(X, y, alpha = a, lambda = cv.fit$lambda.min)
beta.lasso   =     as.vector(fit.ls$beta)
toc()

# fit rid to the whole data set
tic('rid-timing')
a=0 # lasso
cv.fit           =     cv.glmnet(X, y, alpha = a, nfolds = 10)
fit.rid          =     glmnet(X, y, alpha = a, lambda = cv.fit$lambda.min)
beta.rid         =     as.vector(fit.rid$beta)
toc()

#fit random forest to the whole data set
start_time <- Sys.time()
rf.score.all    =  randomForest(Score ~ ., data=df,  mtry= floor(sqrt(p)), importance=TRUE)
# y.hat =  predict(rf.score, newdata = df)
end_time <- Sys.time()
end_time - start_time

# Q5(3) Coefficients(lasso, ridge, elastic-net) and random forest importance of parameters plot
# Q5(3) Use the elastic-net estimated coefficients to create an order based on largest to smallest coefficient

#elastic-net
p=104
betaS.en               =     data.frame(c(1:p), as.vector(fit.en$beta))
colnames(betaS.en)     =     c( "feature", "value")
betaS.en$feature       =     row.names(fit.en$beta)

#lasso
betaS.ls               =     data.frame(c(1:p), as.vector(fit.ls$beta))
colnames(betaS.ls)     =     c( "feature", "value")
betaS.ls$feature       =     row.names(fit.ls$beta)

#rid
betaS.rid              =   data.frame(c(1:p), as.vector(fit.rid$beta))
colnames(betaS.rid)    =     c( "feature", "value")
betaS.rid$feature      =     row.names(fit.rid$beta)

#rf
betaS.rf               =   data.frame(c(1:p), as.vector(rf.score.all$importance[,1]))
colnames(betaS.rf)     =     c( "feature", "value")
betaS.rf$feature       =     row.names(rf.score.all$importance)

# Resort the columns by the elastict-net coefficient value
betaS.ls$feature     =  factor(betaS.ls$feature, levels = betaS.en$feature[order(betaS.en$value, decreasing = TRUE)])
betaS.en$feature     =  factor(betaS.en$feature, levels = betaS.en$feature[order(betaS.en$value, decreasing = TRUE)])
betaS.rid$feature    =  factor(betaS.rid$feature, levels = betaS.rid$feature[order(betaS.en$value,decreasing = TRUE)])
betaS.rf$feature    =  factor(betaS.rf$feature, levels = betaS.rf$feature[order(betaS.rf$value,decreasing = TRUE)])

colnames(betaS.en)[2] <- "en.coef"
colnames(betaS.ls)[2] <- "ls.coef"
colnames(betaS.rid)[2] <- "rid.coef"
colnames(betaS.rf)[2] <- "rf.imp"

merge1 <- merge(betaS.en,betaS.ls,by="feature")
merge2 <- merge(merge1,betaS.rid,by="feature")
merge3 <- merge(merge2,betaS.rf,by="feature")

plot.data <- merge3[order(- merge3$en.coef),]

# Plot 1: Show variable names

enPlot =  ggplot(plot.data[, 1:2], aes(x=feature, y=en.coef)) +
  geom_bar(stat = "identity", fill="tan3", colour="black")+
  ggtitle("Elastic Net Coefficient")+ 
  theme(axis.text.x = element_text(color = "grey20", size = 5, angle = 90, hjust = .5, vjust = .5, face = "plain"), axis.title.x=element_blank())


lsPlot =  ggplot(plot.data[c("feature","ls.coef")], aes(x=feature, y=ls.coef),) +
  geom_bar(stat = "identity", fill="plum2", colour="black")+
  ggtitle("Lasso Coefficient") + 
  theme(axis.text.x = element_text(color = "grey20", size = 5, angle = 90, hjust = .5, vjust = .5, face = "plain"), axis.title.x=element_blank())  


ridPlot=ggplot(plot.data[c("feature","rid.coef")], aes(x=feature, y=rid.coef))+
  geom_bar(stat="identity",fill="slategray1",colour="black")+
  ggtitle("Ridge Coefficient") + 
  theme(axis.text.x = element_text(color = "grey20", size = 5, angle = 90, hjust = .5, vjust = .5, face = "plain"), axis.title.x=element_blank())

rfPlot=ggplot(plot.data[c("feature","rf.imp")], aes(x=feature, y=rf.imp))+
  geom_bar(stat="identity",fill="palegreen",colour="black")+
  ggtitle("Random Forest Importance") + 
  theme(axis.text.x = element_text(color = "grey20", size = 5, angle = 90, hjust = .5, vjust = .5, face = "plain"), axis.title.x=element_blank())

# Generate Plot 1:
grid.arrange(enPlot, lsPlot,ridPlot, rfPlot , ncol=1)


# Plot 2: No Variable Name

enPlot2 =  ggplot(plot.data[, 1:2], aes(x=feature, y=en.coef)) +
  geom_bar(stat = "identity", fill="tan3", colour="black")+
  ggtitle("Elastic Net Coefficient")+ 
  theme(axis.text.x = element_blank(), axis.title.x=element_blank())


lsPlot2 =  ggplot(plot.data[c("feature","ls.coef")], aes(x=feature, y=ls.coef),) +
  geom_bar(stat = "identity", fill="plum2", colour="black")+
  ggtitle("Lasso Coefficient") +
  theme(axis.text.x = element_blank(), axis.title.x=element_blank())


ridPlot2 =ggplot(plot.data[c("feature","rid.coef")], aes(x=feature, y=rid.coef))+
  geom_bar(stat="identity",fill="slategray1",colour="black")+
  ggtitle("Ridge Coefficient") +
  theme(axis.text.x = element_blank(), axis.title.x=element_blank())

rfPlot2 =ggplot(plot.data[c("feature","rf.imp")], aes(x=feature, y=rf.imp))+
  geom_bar(stat="identity",fill="palegreen",colour="black")+
  ggtitle("Random Forest Importance") +
  theme(axis.text.x = element_blank(), axis.title.x=element_blank())

# Generate Plot 2:
grid.arrange( enPlot2, lsPlot2, ridPlot2, rfPlot2,  ncol=1)



