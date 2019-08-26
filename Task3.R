#Task 3 - Sumeet Kumar - 5873137 - SK521

#import libraries
#install.packages("twitteR")
#install.packages(c('ROAuth','RCurl'))
#install.packages("dplyr")
#install.packages('purrr')
#install.packages('caret')
#install.packages('RTextTools')
#install.packages('maxent')
#install.packages('sos')
library(sos)
library(twitteR)
library('ROAuth')
library('RCurl')
library(dplyr)
library(purrr)
library(plyr)
library(stringr)
library(tm)
library(e1071)
library(caret)
library(RTextTools)
library(maxent)

#twitter authentication
consumerAPIKey <- "#"
consumerAPISecretKey <- "#"
accessTokenKey <- "#"
accessTokenSecretKey <- "#"
reqURL <- "https://api.twitter.com/oauth/request_token"
accessURL <- "https://api.twitter.com/oauth/access_token"
authURL <- "https://api.twitter.com/oauth/authorize"


setup_twitter_oauth(consumerAPIKey,consumerAPISecretKey,accessTokenKey,accessTokenSecretKey)

# collect tweets from twitter
userTimelineTweets <- userTimeline("@BillGates", 101)
#userTimelineTweets

#convert to data frame
tl <- twListToDF(userTimelineTweets)

#write to csv file and save the tweets - change file location
write.csv(tl,file="twitterList.csv")


#Stats of the obtained dataset
billGatesResult<- read.csv("tagging.csv")
#glimpse(billGatesResult)
#glimpse(billGatesResult, width = getOption("width"))
#glimpse(billGatesResult$text, width = getOption("width"))
#glimpse(billGatesResult$Tagging, width = getOption("width"))

#getting the classes
countTagging <- table(billGatesResult$Tagging)

#Corpus
corpusVector <- Corpus(VectorSource(billGatesResult$text))


#pre-processing
pre_process_tweets <- tm_map(corpusVector,  content_transformer(tolower))
url <- content_transformer(function(x) gsub("(f|ht)tp(s?)://\\S+", "", x, perl=T))
pre_process_tweets <- tm_map(pre_process_tweets, url)
pre_process_tweets <- tm_map(pre_process_tweets, removePunctuation)
pre_process_tweets <- tm_map(pre_process_tweets, stripWhitespace)
pre_process_tweets <- tm_map(pre_process_tweets, removeWords, stopwords("english"))
pre_process_tweets <- tm_map(pre_process_tweets, removeNumbers)
pre_process_tweets <-tm_map(pre_process_tweets,stemDocument)

#document term matrix
dtm <- DocumentTermMatrix(pre_process_tweets)

#partition train and test data
trainingDataSet<- billGatesResult$Tagging[1:80] #trainig set
testDataSet <-billGatesResult$Tagging[81:101] #test set

#document term matrix TF-IDF
dtmTrain_TFIDF <- DocumentTermMatrix(pre_process_tweets, control = list(weighting = weightTfIdf))
dtmTrainingSet <- dtmTrain_TFIDF[1:80,]
dtmTestSet <- dtmTrain_TFIDF[81:101,]

# use the NB classifier with Laplace smoothing
naiveBayesResult <- naiveBayes(as.matrix(dtmTrainingSet), trainingDataSet, laplace=1)
#naiveBayesResult

# predict with testdata
predictResult <- predict (naiveBayesResult,as.matrix(dtmTestSet))
#predictResult

#Confusion Matrix
xtab <- table( "Actual" = testDataSet, "Predictions"= predictResult)
#xtab
confMatrixNB <- confusionMatrix(xtab)
confMatrixNB


#apply svm and confusion matrix matrix
trainSVM <- svm(as.matrix(dtmTrainingSet), trainingDataSet, type='C-classification',kernel="radial", cost=10, gamma=0.5)
trainSVM
#Prediction Result
predSVM <- predict(trainSVM, as.matrix(dtmTestSet))
#predSVM

#Confusion Matrix
xSVM <- table("Actual" = testDataSet, "Predictions"= predSVM)
#xSVM
confMatrixSVM <- confusionMatrix(xSVM)
confMatrixSVM


#Performance for Naive Bayes - Precison, Recall, Fmeasure
n1 = sum(xtab) # number of instances
nc1 = nrow(xtab) # number of classes
diag1 = diag(xtab) # number of correctly classified instances per class 
rowsums1 = apply(xtab, 1, sum) # number of instances per class
colsums1 = apply(xtab, 2, sum) # number of predictions per class
p1 = rowsums1 / n1 # distribution of instances over the actual classes
q1 = colsums1 / n1 # distribution of instances over the predicted classes

accuracy1 = sum(diag1) / n1 

precisionNB = diag1 / colsums1 
recallNB = diag1 / rowsums1 
f1NB = 2 * precisionNB * recallNB / (precisionNB + recallNB)
data.frame(precisionNB, recallNB, f1NB, accuracy1) 



#Performance for SVM - Precison, recall, fmeasure

n2 = sum(xSVM) # number of instances
nc2 = nrow(xSVM) # number of classes
diag2 = diag(xSVM) # number of correctly classified instances per class 
rowsums2 = apply(xSVM, 1, sum) # number of instances per class
colsums2 = apply(xSVM, 2, sum) # number of predictions per class
p2 = rowsums2 / n2 # distribution of instances over the actual classes
q2 = colsums2 / n2 # distribution of instances over the predicted classes

accuracy2 = sum(diag2) / n2 

precisionSVM = diag2 / colsums2 
recallSVM = diag2 / rowsums2 
f1SVM = 2 * precisionSVM * precallSVM / (precisionSVM + precisionSVM) 
data.frame(precisionSVM, recallSVM, f1SVM, accuracy2) 