library(klaR)
library(caret)


wdat <- read.table("Desktop/Applied Machine Learning/CS498AppliedMachineLearning/PrimaIndians/pima-indians-diabetes.data", sep = ",")
# Source: http://archive.ics.uci.edu/ml/machine-learning-databases/pima-indians-diabetes/
# Col names of the data set
# 1. Number of times pregnant
# 2. Plasma glucose concentration a 2 hours in an oral glucose tolerance test
# 3. Diastolic blood pressure (mm Hg)
# 4. Triceps skin fold thickness (mm)
# 5. 2-Hour serum insulin (mu U/ml)
# 6. Body mass index (weight in kg/(height in m)^2)
# 7. Diabetes pedigree function
# 8. Age (years)
# 9. Class variable (0 or 1)

bigx<-wdat[,-c(9)]
bigy<-wdat[,9]
trscore <- array(dim = 10)
tescore <- array(dim = 10)


#####################
#Part A
#############
for (wi in 1:10) {
  wtd <- createDataPartition(y = bigy, p = 0.8, list = FALSE) # 80% of the data into training
  nbx <- bigx                                 # matrix of features
  ntrbx <- nbx[wtd, ]                         # training features
  ntrby <- bigy[wtd]                          # training labels
  
  trposflag <- ntrby > 0                      # training labels for diabetes positive
  ptregs <- ntrbx[trposflag, ]                # training rows features with diabetes positive
  ntregs <- ntrbx[!trposflag, ]               # training rows features with diabetes negative
  
  ntebx <- nbx[-wtd, ]                        # test rows - features
  nteby <- bigy[-wtd]                         # test rows - labels
  
  ptrmean <- sapply(ptregs, mean, na.rm = T)  # vector of means for training, diabetes positive
  ntrmean <- sapply(ntregs, mean, na.rm = T)  # vector of means for training, diabetes negative
  ptrsd   <- sapply(ptregs, sd, na.rm = T)    # vector of sd for training, diabetes positive
  ntrsd   <- sapply(ntregs, sd, na.rm = T)    # vector of sd for training, diabetes negative
  
  ptroffsets <- t(t(ntrbx) - ptrmean)         # first step normalize training diabetes pos, subtract mean
  ptrscales  <- t(t(ptroffsets) / ptrsd)      # second step normalize training diabetes pos, divide by sd
  ptrlogs    <- -(1/2) * rowSums(apply(ptrscales, c(1,2),
                function(x) x^2), na.rm = T) - sum(log(ptrsd))+log(NROW(ptregs)/NROW(ntrby))  # Log likelihoods based on 
								# normal distr. for diabetes positive
  ntroffsets <- t(t(ntrbx) - ntrmean)
  ntrscales  <- t(t(ntroffsets) / ntrsd)
  ntrlogs    <- -(1/2) * rowSums(apply(ntrscales, c(1,2) 
                                       , function(x) x^2), na.rm = T) - sum(log(ntrsd)) +log(NROW(ntregs)/NROW(ntrby))
                                                                # Log likelihoods based on 
								# normal distr for diabetes negative
                                                                # (It is done separately on each class)
  
  lvwtr      <- ptrlogs > ntrlogs              # Rows classified as diabetes positive by classifier 
  gotrighttr <- lvwtr == ntrby                 # compare with true labels
  trscore[wi]<- sum(gotrighttr)/(sum(gotrighttr)+sum(!gotrighttr)) # Accuracy with training set
  
  pteoffsets <- t(t(ntebx)-ptrmean)            # Normalize test dataset with parameters from training
  ptescales  <- t(t(pteoffsets)/ptrsd)
  ptelogs    <- -(1/2)*rowSums(apply(ptescales,c(1, 2)
                                     , function(x)x^2), na.rm=TRUE)-sum(log(ptrsd)) +log(NROW(ptregs)/NROW(ntrby))
  
  nteoffsets <- t(t(ntebx)-ntrmean)            # Normalize again for diabetes negative class
  ntescales  <- t(t(nteoffsets)/ntrsd)
  ntelogs    <- -(1/2)*rowSums(apply(ntescales,c(1, 2)
                                     , function(x)x^2), na.rm=TRUE)-sum(log(ntrsd)) +log(NROW(ntregs)/NROW(ntrby))
  
  lvwte<-ptelogs>ntelogs
  gotright<-lvwte==nteby
  tescore[wi]<-sum(gotright)/(sum(gotright)+sum(!gotright))  # Accuracy on the test set
}
print(tescore)
print(mean(tescore))

#########
#Part B
########

wdat2 <-wdat
#replace 0 to NA
wdat2$V3[wdat2$V3 == 0] <- NA
wdat2$V4[wdat2$V4 == 0] <- NA
wdat2$V6[wdat2$V6 == 0] <- NA
wdat2$V8[wdat2$V8 == 0] <- NA

bigx<-wdat2[,-c(9)]
bigy<-wdat2[,9]
trscore <- array(dim = 10)
tescore <- array(dim = 10)


for (wi in 1:10) {
  wtd <- createDataPartition(y = bigy, p = 0.8, list = FALSE) # 80% of the data into training
  nbx <- bigx                                 # matrix of features
  ntrbx <- nbx[wtd, ]                         # training features
  ntrby <- bigy[wtd]                          # training labels
  
  trposflag <- ntrby > 0                      # training labels for diabetes positive
  ptregs <- ntrbx[trposflag, ]                # training rows features with diabetes positive
  ntregs <- ntrbx[!trposflag, ]               # training rows features with diabetes negative
  
  ntebx <- nbx[-wtd, ]                        # test rows - features
  nteby <- bigy[-wtd]                         # test rows - labels
  
  ptrmean <- sapply(ptregs, mean, na.rm = T)  # vector of means for training, diabetes positive
  ntrmean <- sapply(ntregs, mean, na.rm = T)  # vector of means for training, diabetes negative
  ptrsd   <- sapply(ptregs, sd, na.rm = T)    # vector of sd for training, diabetes positive
  ntrsd   <- sapply(ntregs, sd, na.rm = T)    # vector of sd for training, diabetes negative
  
  ptroffsets <- t(t(ntrbx) - ptrmean)         # first step normalize training diabetes pos, subtract mean
  ptrscales  <- t(t(ptroffsets) / ptrsd)      # second step normalize training diabetes pos, divide by sd
  ptrlogs    <- -(1/2) * rowSums(apply(ptrscales, c(1,2),
                function(x) x^2), na.rm = T) - sum(log(ptrsd))+log(NROW(ptregs)/NROW(ntrby))  # Log likelihoods based on 
								# normal distr. for diabetes positive
  ntroffsets <- t(t(ntrbx) - ntrmean)
  ntrscales  <- t(t(ntroffsets) / ntrsd)
  ntrlogs    <- -(1/2) * rowSums(apply(ntrscales, c(1,2) 
                                       , function(x) x^2), na.rm = T) - sum(log(ntrsd)) +log(NROW(ntregs)/NROW(ntrby))
                                                                # Log likelihoods based on 
								# normal distr for diabetes negative
                                                                # (It is done separately on each class)
  
  lvwtr      <- ptrlogs > ntrlogs              # Rows classified as diabetes positive by classifier 
  gotrighttr <- lvwtr == ntrby                 # compare with true labels
  trscore[wi]<- sum(gotrighttr)/(sum(gotrighttr)+sum(!gotrighttr)) # Accuracy with training set
  
  pteoffsets <- t(t(ntebx)-ptrmean)            # Normalize test dataset with parameters from training
  ptescales  <- t(t(pteoffsets)/ptrsd)
  ptelogs    <- -(1/2)*rowSums(apply(ptescales,c(1, 2)
                                     , function(x)x^2), na.rm=TRUE)-sum(log(ptrsd)) +log(NROW(ptregs)/NROW(ntrby))
  
  nteoffsets <- t(t(ntebx)-ntrmean)            # Normalize again for diabetes negative class
  ntescales  <- t(t(nteoffsets)/ntrsd)
  ntelogs    <- -(1/2)*rowSums(apply(ntescales,c(1, 2)
                                     , function(x)x^2), na.rm=TRUE)-sum(log(ntrsd)) +log(NROW(ntregs)/NROW(ntrby))
  
  lvwte<-ptelogs>ntelogs
  gotright<-lvwte==nteby
  tescore[wi]<-sum(gotright)/(sum(gotright)+sum(!gotright))  # Accuracy on the test set
}
print(tescore)
print(mean(tescore))


###########
#Part C
###########

##Model with Klar
trainPartition <- createDataPartition(y = wdat[,9], p = 0.8, list = FALSE)
train <- wdat[trainPartition,]
test <-wdat[-trainPartition,]
model1 <- klaR::NaiveBayes(as.factor(V9)~.,data= train )
pred1 <- predict(object = model1, test[,-c(9)])
gotright = test[,9] ==pred1$class
accuracy =  sum(gotright)/(sum(gotright)+sum(!gotright))


#Model with caret
model2 <- caret::train(x = train[,-c(9)], y= as.factor(train[,c(9)]),"nb", trControl=trainControl(method='cv',number=10)  )
pred2 <- predict(model2, test[,-c(9)])
gotright = test[,9] ==pred2
accuracy =  sum(gotright)/(sum(gotright)+sum(!gotright))

#########
#Part D
##########
#SVM Model
setwd("Desktop/Applied Machine Learning/svm_light_osx.8.4_i7/")
svmModel <-  svmlight(V9 ~ ., data=train, pathsvm='../../svm_light_osx.8.4_i7')
pred3 <- predict(svmModel, test[,-c(9)])
gotright = test[,9] ==pred3$class
accuracy =  sum(gotright)/(sum(gotright)+sum(!gotright))








