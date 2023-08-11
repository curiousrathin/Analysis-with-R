install.packages("tidyverse")
library(tidyverse)
install.packages("InformationValue")
library(party)
install.packages("RWeka")
library(RWeka)
install.packages("partykit")
library(partykit)
install.packages('class')
library(class)
install.packages("InformationValue")
library(InformationValue)
install.packages("caret")
library(caret)
install.packages("gmodels")
library(gmodels)
install.packages("party")
library(party)
install.packages("ISLR")
library(ISLR)
setwd("~/Desktop/ITM618")
getwd()


BankData <- read.csv("bank.csv", sep = ",", header = TRUE)
View(BankData)
#Separate into multiple columns and removing poutcome column 
BankData <- BankData %>%
  separate(age.job.marital.education.default.balance.housing.loan.contact.day.month.duration.campaign.pdays.previous.poutcome.y, into = c('age','job','marital','education','default','balance','housing','loan','contact','day','month','duration','campaign','pdays','previous','poutcome','y'), ";") %>%
  select(age, job, marital, education, default, balance, housing, loan, contact, campaign, pdays, previous, y)
dim(BankData)
#Description of the given dataset
dim(BankData)
sapply(BankData, class)
BankData$age <- as.numeric(BankData$age)
BankData$balance <- as.numeric(BankData$balance)
BankData[sapply(BankData, is.character)] <- lapply(BankData[sapply(BankData, is.character)], as.factor)
sapply(BankData, class)

#Code to get the statistics of the given dataset 
#Statistics of Given Dataset without the ‘poutcome’
summary(BankData)
#Boxplot
boxplot(age~y, data = BankData, main="Age Statistics of Subscribers", xlab="Has the client subscribed to a term deposit", ylab="Age")
boxplot(campaign~y, data = BankData, main="Campaign Statistics of Subscribers", xlab="Has the client subscribed to a term deposit", ylab="No. of Campaigns")
#Statistics of Unknown Entries
Unknown_Entries<- filter(BankData, job == "unknown" | education == "unknown" | contact == "unknown")
dim(Unknown_Entries)
summary(Unknown_Entries)
#Percentage of Unknown Entries
NROW(Unknown_Entries)/NROW(BankData) *100
#Code to Data Cleaning
#Remove rows that have unknown 
New_BankData <- BankData[!(BankData$job=="unknown" | BankData$education=="unknown" | BankData$contact=="unknown"),]
summary(New_BankData)
dim(New_BankData)
#Reviewing metrics for classifiers
barplot(table(New_BankData$y),main="Bank Term Deposit Subscriptions", xlab="Subscribed?", ylab="# of Subscriptions")
barplot(table(New_BankData$y, New_BankData$age), main="Subscriptions by Age", xlab="Age", ylab="Subscriptions")
#Code to get the statistics of the cleaned dataset
summary(New_BankData)
dim(New_BankData)
#Saving the final data set the classifiers
write.csv(New_BankData, "New_BankData.csv", row.names = FALSE)
#Code to get the statistics of the cleaned dataset
summary(New_BankData)
dim(New_BankData)
#Code to Prepare the Training and the Test Dataset
AllData <- read.csv("New_BankData.csv", header = TRUE, stringsAsFactors = FALSE)
set.seed(123)
myIndex <- sample(1:nrow(AllData), 0.8*nrow(AllData))
#Creating training data
train_BankData <- AllData[myIndex, ]
summary(train_BankData)
dim(train_BankData)
#Creating test data
test_BankData <- AllData[-myIndex, ]
summary(test_BankData)
dim(test_BankData)

#Saving training and test data into csv files
write.csv(train_BankData, "bankdata_training.csv", row.names = FALSE)
write.csv(test_BankData, "bankdata_test.csv", row.names = FALSE)
#Code to CTree classification and to generate Confusion Matrix for CTree
####CTree Classification
train_BankData_CTree <- train_BankData
train_BankData_CTree <- train_BankData_CTree[ -c(1,6,10:12) ]
test_BankData_CTree <- test_BankData
test_BankData_CTree <- test_BankData_CTree[ -c(1,6,10:12) ]
train_BankData_CTree[] <- lapply(train_BankData_CTree, factor)
test_BankData_CTree[] <- lapply(test_BankData_CTree, factor)
#use all attributes to predict the value of the target class
formula_all_ctree <- y ~ .
#use two selected attributes to predict the value of the target class
formula_selected_ctree <- y ~ job + housing
#Calculate Information Gain
weights <- InfoGainAttributeEval(formula_all_ctree, data = train_BankData_CTree)
barplot(weights, las=0)
#R builds trees using both numeric and categorical data
#CTree with all attributes
CTree_all <- ctree(formula_all_ctree, data = train_BankData_CTree)
plot(CTree_all)
testCTree_all <- predict(CTree_all, newdata = test_BankData_CTree)
table(testCTree_all, test_BankData_CTree$y)

table(predict(CTree_all, newdata = test_BankData_CTree), test_BankData_CTree$y)
#CTree with selected attributes
CTree_selected <- ctree(formula_selected_ctree, data = train_BankData_CTree)
plot(CTree_selected)
testCTree_selected <- predict(CTree_selected, newdata = test_BankData_CTree)
table(testCTree_selected, test_BankData_CTree$y)
CTree_train <- ctree(formula_all_ctree, data = train_BankData_CTree)
pred_ctree <- predict(CTree_all, newdata = test_BankData_CTree)
actual_ctree <- test_BankData_CTree$y
TP_ctree <- sum(actual_ctree == 'yes' & pred_ctree == 'yes')
FP_ctree <- sum(actual_ctree == 'no' & pred_ctree == 'yes')
TN_ctree <- sum(actual_ctree == 'no' & pred_ctree == 'no')
FN_ctree <- sum(actual_ctree == 'yes' & pred_ctree == 'no')
#Confusion Matrix
cm_ctree<- data.frame("Confusion Matrix" = c("Actual Yes", "Actual No"),
                      "Predicted Yes" = c(TP_ctree, FP_ctree),
                      "Predicted No" = c(FN_ctree, TN_ctree))
print(cm_ctree)
confusionMatrix(table(pred_ctree, actual_ctree))
CrossTable(x = actual_ctree, y = pred_ctree, prop.chisq = FALSE)

#Code to J48 classification and to generate Confusion Matrix for J48
#J48 Classification
myIndex <- sample(1:nrow(AllData), 0.8*nrow(AllData))
train_BankData <- AllData[myIndex, ]
test_BankData <- AllData[-myIndex, ]
train_BankData_J48 <- train_BankData
train_BankData_J48 <- train_BankData_J48[ -c(1,6,10:12) ]
test_BankData_J48 <- test_BankData
test_BankData_J48 <- test_BankData_J48[ -c(1,6,10:12) ]
train_BankData_J48[] <- lapply(train_BankData_J48, factor)
test_BankData_J48[] <- lapply(test_BankData_J48, factor)
#use all attributes to predict the value of the target class
formula_all_j48 <- y ~ .
#use two selected attributes to predict the value of the target class
formula_selected_j48 <- y ~ job + housing
#Calculate Information Gain
weights <- InfoGainAttributeEval(formula_all_j48, data = train_BankData_J48)
barplot(weights, las=0)
#R builds trees using both numeric and categorical data
#J48 with all attributes
J48_all <- J48(formula_all_j48, data = train_BankData_J48)
plot(J48_all)
testJ48_all <- predict(J48_all, newdata = test_BankData_J48)
table(testJ48_all, test_BankData_J48$y)
table(predict(J48_all, newdata = test_BankData_J48), test_BankData_J48$y)
#J48 with selected attributes
J48_selected <- J48(formula_selected_j48, data = train_BankData_J48)
plot(J48_selected)
testJ48_selected <- predict(J48_selected, newdata = test_BankData_J48)
table(testJ48_selected, test_BankData_J48$y)
J48_train <- J48(formula_all_j48, data = train_BankData_J48)
pred_J48 <- predict(J48_all, newdata = test_BankData_J48)
actual_J48 <- test_BankData_J48$y
TP_J48 <- sum(actual_J48 == 'yes' & pred_J48 == 'yes')
FP_J48 <- sum(actual_J48 == 'no' & pred_J48 == 'yes')
TN_J48 <- sum(actual_J48 == 'no' & pred_J48 == 'no')
FN_J48 <- sum(actual_J48 == 'yes' & pred_J48 == 'no')
#Confusion Matrix
cm_J48<- data.frame("Confusion Matrix" = c("Actual Yes", "Actual No"),
                    "Predicted Yes" = c(TP_J48, FP_J48),
                    "Predicted No" = c(FN_J48, TN_J48))
print(cm_J48)
confusionMatrix(table(pred_J48, actual_J48))



#Code to KNN classification and to generate Confusion Matrix for KNN
###K-NN Classification
myIndex_KNN <- sample(1:nrow(AllData), 0.8*nrow(AllData))
train_BankData <- AllData[myIndex_KNN, ]
test_BankData <- AllData[-myIndex_KNN, ]
#Training Data
train_BankData_KNN <- train_BankData
train_BankData_KNN <- train_BankData_KNN %>%
  select(age,campaign, balance, pdays, previous,y)
train_BankData_KNN$y <- factor(train_BankData_KNN$y, levels = c("no", "yes"), labels = c("No", "Yes"))

#Test Data
test_BankData_KNN <- test_BankData
test_BankData_KNN <- test_BankData_KNN %>%
  select(age,campaign, balance, pdays, previous,y)
test_BankData_KNN <- as.data.frame(test_BankData_KNN)
test_BankData_KNN$y <- factor(test_BankData_KNN$y, levels = c("no", "yes"), labels = c("No", "Yes"))

#Normalizing the numerical variables in both training and test data
train_Cols_Norm <- train_BankData_KNN %>%
  select(age,campaign, balance, pdays, previous)
normalize <- function(x) {
  return ((x - min(x)) / (max(x) - min(x))) }
train_Cols_Norm <- as.data.frame(lapply(train_Cols_Norm, normalize))

test_Cols_Norm <- test_BankData_KNN %>%
  select(age,campaign, balance, pdays, previous)
normalize <- function(x) {
  return ((x - min(x)) / (max(x) - min(x))) }
test_Cols_Norm <- as.data.frame(lapply(test_Cols_Norm, normalize))

#Combining the two data frames
train_BankData_KNN2 <- cbind(train_BankData_KNN, train_Cols_Norm)
test_BankData_KNN2 <- cbind(test_BankData_KNN, test_Cols_Norm)
#Removing non-essential columns for k-nn classification
train_BankData_KNN <- train_BankData_KNN2[ -c(1:5) ]
summary(train_BankData_KNN)
test_BankData_KNN <- test_BankData_KNN2[ -c(1:5) ]
summary(test_BankData_KNN)

#Find the number of observation to get the value for k
Total.Obs <- NROW(train_BankData_KNN$y) 
print(Total.Obs)
#Training a model on data
K.145 = sqrt(Total.Obs)
print(K.145)
KNN.145_pred <- knn(train = train_BankData_KNN[c(2:6)], test = test_BankData_KNN[c(2:6)], cl = train_BankData_KNN$y, k=K.145)

#Model Evaluation
ACC.145 <- 100 * sum(test_BankData_KNN$y == KNN.145_pred)/NROW(test_BankData_KNN$y)
#Table form for accuracy
table(KNN.145_pred,test_BankData_KNN$y)
barplot(table(KNN.145_pred ,test_BankData_KNN$y))
#Print accuracy 
print(ACC.145)
#Confusion Matrix

CrossTable(x = test_BankData_KNN$y, y = KNN.145_pred, prop.chisq = FALSE)
#Confusion Matrix
confusionMatrix(table(KNN.145_pred, test_BankData_KNN$y))

#Code to Linear classification and to generate Confusion Matrix for Linear Classifier
###Linear Classification
myIndex_linear <- sample(1:nrow(AllData), 0.8*nrow(AllData))


##Creating training and test data
train_BankData <- AllData[myIndex_linear, ]
test_BankData <- AllData[-myIndex_linear, ]

train_BankData_linear <- train_BankData
test_BankData_linear <- test_BankData

train_BankData_linear$y <- as.factor(train_BankData_linear$y)
test_BankData_linear$y <- as.factor(test_BankData_linear$y)
## use all atributes to predict the value of the target class
formula_all_linear <- y ~ .

linear_clf <- glm(formula_all_linear, data = train_BankData_linear, family = 'binomial') 
summary(linear_clf)
# Testing/Prediction
pred_val <- predict(linear_clf, test_BankData_linear, type = "response") 
str(pred_val)
# use threshold given
threshold <- 1.5 
pred_linear <- ifelse(pred_val > threshold, "yes", "no") 
actual_linear <- test_BankData_linear$y 
str(test_BankData_linear$y)

# Compute True Positives
tp_linear <- sum(actual_linear =='yes' &pred_linear =='yes')
# False Positives
fp_linear <- sum(actual_linear =='no' &pred_linear =='yes')
# True Negatives
tn_linear <- sum(actual_linear =='no' &pred_linear =='no')
# False Negatives
fn_linear <- sum(actual_linear =='yes' &pred_linear =='no') 

cm_linear<- data.frame("Confusion Matrix of Linear" = c("Actual Yes", "Actual No"),
                       "Predicted Yes" = c(tp_linear, fp_linear),
                       "Predicted No" = c(fn_linear, tn_linear))

print(cm_linear)

#Code to Generate the Bar charts of 5.1 to 5.5 
### Classifier Comparison
Classifiers <- data.frame("Classifier Comparison" = c("CTree", "J48", "Linear Classification", "k-NN"),
                          "Accuracy" = c(0.8513, 0.8507,0.8538,0.8598),
                          "Error_Rate" = c(0.1487,0.1285,0.1462,0.1402),
                          "Precision" = c(0,0,0,0.4231),
                          "Recall" = c(0,0,0,0.0127),
                          "F1_Score" = c(0,0,0,0.0246))
print(Classifiers)
barplot(Classifiers$Accuracy, main = "Classifier Comparison Based on Accuracy",
        names.arg = c("CTree", "J48", "Linear Classification", "k-NN"), 
        ylab ="Accuracy", xlab="Classifiers", col="black")

barplot(Classifiers$Error_Rate, main = "Classifier Comparison Based on Error Rate",
        names.arg = c("CTree", "J48", "Linear Classification", "k-NN"), 
        ylab ="Error_Rate", xlab="Classifiers", col="black")
barplot(Classifiers$Precision, main = "Classifier Comparison Based on Precision",
        names.arg = c("CTree", "J48", "Linear Classification", "k-NN"), 
        ylab ="Precision", xlab="Classifiers", col="black")
barplot(Classifiers$Recall, main = "Classifier Comparison Based on Recall",
        names.arg = c("CTree", "J48", "Linear Classification", "k-NN"), 
        ylab ="Recall", xlab="Classifiers", col="black")
barplot(Classifiers$F1_Score, main = "Classifier Comparison Based on F1-Score",
        names.arg = c("CTree", "J48", "Linear Classification", "k-NN"), 
        ylab ="F1_Score", xlab="Classifiers", col="black")


 


