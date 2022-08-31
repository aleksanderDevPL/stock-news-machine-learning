libraries <- c("caret", "quanteda", "e1071", "ggplot2", 
               "irlba", "randomForest", "dplyr", "readxl", "randomForest", "doSNOW",
               "lsa")
lapply(libraries, require, character.only = TRUE)

# load the dataset
dataset <- read_excel("PATH-TO-DATASET")

# tidying dataset
dataset <- dataset[, c(2,4)]
names(dataset) <- c("Text", "Label")
dataset$Label <- as.factor(dataset$Label)

# check row completeness 
length(which(!complete.cases(dataset)))

# check distibution of the class labels (kit vs hit).
prop.table(table(dataset$Label))

#new feature - text length
dataset$TextLength <- nchar(dataset$Text)
summary(dataset$TextLength)

# first check if there are any features that can be identified graphicall
ggplot(dataset, aes(x = TextLength, fill = Label)) +
  theme_bw() +
  geom_histogram(binwidth = 5) +
  labs(y = "Text Count", x = "Length of Text",
       title = "Distribution of Text Lengths with Class Labels")

# split traning 70% / test 30%
set.seed(32911)
indexes <- createDataPartition(dataset$Label, times = 1,
                               p = 0.7, list = FALSE)

# filter out training and test rows
train <- dataset[indexes,]
test <- dataset[-indexes,]

# proportions verification
prop.table(table(train$Label))
prop.table(table(test$Label))

# tokenization
train.tokens <- tokens(train$Text, what = "word", 
                       remove_numbers = TRUE, remove_punct = TRUE,
                       remove_symbols = TRUE, remove_hyphens = TRUE)

train.tokens <- tokens_tolower(train.tokens)
# check on random row result after tokenization
train.tokens[[5]]

# remove polish stopwords
pl_stop_words <- read_lines("C:/Olek/Olek skrypty/polish_stopwords.txt")
pl_stop_words_split <- strsplit(pl_stop_words, ", ")[[1]]
train.tokens <- tokens_select(train.tokens, pl_stop_words_split, 
                              selection = "remove")
# another check on random row result after tokenization
train.tokens[[5]]

# stemming of tokens not possible yet using tokens_wordstem
# hence I've not included that 

# bag-of-words 
train.tokens.dfm <- dfm(train.tokens, tolower = FALSE)

# transformation to matrix
train.tokens.matrix <- as.matrix(train.tokens.dfm)
dim(train.tokens.matrix)

train.tokens.df <- cbind(Label = train$Label, data.frame(train.tokens.dfm))

# cross validation, creation of 30 random stratified samples
# seed to allow the users achieve the same result
set.seed(49701)
cv.folds <- createMultiFolds(train$Label, k = 10, times = 3)

cv.cntrl <- trainControl(method = "repeatedcv", number = 10,
                         repeats = 3, index = cv.folds)


# Create a cluster to optimize the computation
# number of processors in the first argument depends on 
# how many cores the machine has
cl <- makeCluster(3, type = "SOCK")
registerDoSNOW(cl)

# calculate using decision tree model
rpart.cv.1 <- train(Label ~ ., data = train.tokens.df, method = "rpart", 
                    trControl = cv.cntrl, tuneLength = 7)

stopCluster(cl)

# check the results.
rpart.cv.1


#######################################################################################################

# check on the Term Frequency-Inverse Document Frequency (TF-IDF) 
# to reduce impact of repeating tokens in texts that are relatively long
term.freq <- function(row) {
  row / sum(row)
}

inverse.doc.freq <- function(col) {
  corpus.size <- length(col)
  doc.count <- length(which(col > 0))
  
  log10(corpus.size / doc.count)
}

tf.idf <- function(x, idf) {
  x * idf
}
# normalization of texts
train.tokens.df <- apply(train.tokens.matrix, 1, term.freq)
dim(train.tokens.df)

# IDF vector
train.tokens.idf <- apply(train.tokens.matrix, 2, inverse.doc.freq)
str(train.tokens.idf)

# TF-IDF
train.tokens.tfidf <-  apply(train.tokens.df, 2, tf.idf, idf = train.tokens.idf)

train.tokens.tfidf <- t(train.tokens.tfidf)
dim(train.tokens.tfidf)

# prepare for model calculation
train.tokens.tfidf.df <- cbind(Label = train$Label, data.frame(train.tokens.tfidf))
names(train.tokens.tfidf.df) <- make.names(names(train.tokens.tfidf.df))

cl <- makeCluster(3, type = "SOCK")
registerDoSNOW(cl)

# calculate using decision tree model just like in the first case
rpart.cv.2 <- train(Label ~ ., data = train.tokens.tfidf.df, method = "rpart", 
                    trControl = cv.cntrl, tuneLength = 7)

stopCluster(cl)

# check on the results
rpart.cv.2

#####################################################################################################

# addition of bigrams
train.tokens <- tokens_ngrams(train.tokens, n = 1:2)
train.tokens[[5]]

train.tokens.dfm <- dfm(train.tokens, tolower = FALSE)
train.tokens.matrix <- as.matrix(train.tokens.dfm)
train.tokens.dfm

# normalization of text
train.tokens.df <- apply(train.tokens.matrix, 1, term.freq)

# IDF vector
train.tokens.idf <- apply(train.tokens.matrix, 2, inverse.doc.freq)

# TF-IDF 
train.tokens.tfidf <-  apply(train.tokens.df, 2, tf.idf, 
                             idf = train.tokens.idf)

train.tokens.tfidf <- t(train.tokens.tfidf)

train.tokens.tfidf.df <- cbind(Label = train$Label, data.frame(train.tokens.tfidf))
names(train.tokens.tfidf.df) <- make.names(names(train.tokens.tfidf.df))


cl <- makeCluster(3, type = "SOCK")
registerDoSNOW(cl)

# single decision tree model just like in the cases before
rpart.cv.3 <- train(Label ~ ., data = train.tokens.tfidf.df, method = "rpart", 
                     trControl = cv.cntrl, tuneLength = 7)
stopCluster(cl)
 
# check the results
rpart.cv.3

#######################################################################################################

# adding singular value decomposition
# I'm reducing number of columns to 200
# to allow my machine carry on the random forest computations
train.irlba <- irlba(t(train.tokens.tfidf), nv = 75, maxit = 200)

# necessary to perform algebra on the vectors
sigma.inverse <- 1 / train.irlba$d
u.transpose <- t(train.irlba$u)
document <- train.tokens.tfidf[1,]
document.hat <- sigma.inverse * u.transpose %*% document

train.svd <- data.frame(Label = train$Label, train.irlba$v)

cl <- makeCluster(3, type = "SOCK")
registerDoSNOW(cl)

# once again calculate the model using single decision tree
rpart.cv.4 <- train(Label ~ ., data = train.svd, method = "rpart", 
                    trControl = cv.cntrl, tuneLength = 7)

stopCluster(cl)

# check on the results
rpart.cv.4



#################################################################################################
# calculation using random forest

cl <- makeCluster(3, type = "SOCK")
registerDoSNOW(cl)

# using default number of trees is 500
rf.cv.1 <- train(Label ~ ., data = train.svd, method = "rf", 
                 trControl = cv.cntrl, tuneLength = 7)

stopCluster(cl)

# check on the results
rf.cv.1
confusionMatrix(train.svd$Label, rf.cv.1$finalModel$predicted)

#########################################################################################################

# add text length feature and recompute random forest 
train.svd$TextLength <- train$TextLength

cl <- makeCluster(3, type = "SOCK")
registerDoSNOW(cl)

rf.cv.2 <- train(Label ~ ., data = train.svd, method = "rf",
                 trControl = cv.cntrl, tuneLength = 7, 
                 importance = TRUE)

stopCluster(cl)

rf.cv.2
confusionMatrix(train.svd$Label, rf.cv.2$finalModel$predicted)


# check on the importance of new feature
varImpPlot(rf.cv.1$finalModel)
varImpPlot(rf.cv.2$finalModel)


###################################################################################################

# adding cosine similarity as new feature
# hypothetis being the text marked as 'hit' are similar to each other
# recompute using random forest

train.similarities <- cosine(t(as.matrix(train.svd[, -c(1, ncol(train.svd))])))
hit.indexes <- which(train$Label == "hit")
train.svd$HitSimilarity <- rep(0.0, nrow(train.svd))
for(i in 1:nrow(train.svd)) {
  train.svd$HitSimilarity[i] <- mean(train.similarities[i, hit.indexes])  
}

cl <- makeCluster(3, type = "SOCK")
registerDoSNOW(cl)

rf.cv.3 <- train(Label ~ ., data = train.svd, method = "rf",
                 trControl = cv.cntrl, tuneLength = 7,
                 importance = TRUE)

stopCluster(cl)

rf.cv.3
confusionMatrix(train.svd$Label, rf.cv.3$finalModel$predicted)
varImpPlot(rf.cv.3$finalModel)

#################################################################################################

# verify the models using test data subset

test.tokens <- tokens(test$Text, what = "word", 
                      remove_numbers = TRUE, remove_punct = TRUE,
                      remove_symbols = TRUE, remove_hyphens = TRUE)
test.tokens <- tokens_tolower(test.tokens)
test.tokens <- tokens_select(test.tokens, pl_stop_words_split, 
                             selection = "remove")

test.tokens <- tokens_ngrams(test.tokens, n = 1:2)
test.tokens.dfm <- dfm(test.tokens, tolower = FALSE)
test.tokens.dfm <- dfm_select(test.tokens.dfm, pattern = train.tokens.dfm,
                              selection = "keep")
test.tokens.matrix <- as.matrix(test.tokens.dfm)
test.tokens.df <- apply(test.tokens.matrix, 1, term.freq)
str(test.tokens.df)
test.tokens.tfidf <-  apply(test.tokens.df, 2, tf.idf, idf = train.tokens.idf)
dim(test.tokens.tfidf)
test.tokens.tfidf <- t(test.tokens.tfidf)
summary(test.tokens.tfidf[1,])
test.tokens.tfidf[is.na(test.tokens.tfidf)] <- 0.0
summary(test.tokens.tfidf[1,])

test.svd.raw <- t(sigma.inverse * u.transpose %*% t(test.tokens.tfidf))
test.svd <- data.frame(Lab el = test$Label, test.svd.raw, 
                       TextLength = test$TextLength)

test.similarities <- rbind(test.svd.raw, train.irlba$v[hit.indexes,])
test.similarities <- cosine(t(test.similarities))
test.svd$HitSimilarity <- rep(0.0, nrow(test.svd))
hit.cols <- (nrow(test.svd) + 1):ncol(test.similarities)
for(i in 1:nrow(test.svd)) {
  test.svd$HitSimilarity[i] <- mean(test.similarities[i, hit.cols])  
}
test.svd$HitSimilarity[!is.finite(test.svd$HitSimilarity)] <- 0

# make prediction on test data using model that I've 
# achieved the best results with
preds <- predict(rf.cv.3, test.svd)
# check the results
confusionMatrix(preds, test.svd$Label)
