dev.off()
rm(list=ls())

library(ROCR)
library(neuralnet)
library(REdaS)
library(tibble)
library(dplyr) 
library(caret)
library(DMwR)
library(ROSE)
library(factoextra)
library(xgboost)
library(magrittr)
library(Matrix)
library(rpart)
library(regclass)
library(randomForest)
library("Ckmeans.1d.dp") # for xgb.ggplot.importance


set.seed(2018)

m <- read.csv("/Users/Raghav/Desktop/RxM/MSc DA/Semester 2/ADM (H9ADM)/Project/Dataset/exoTrain.csv", stringsAsFactors = F)
n = read.csv("/Users/Raghav/Desktop/RxM/MSc DA/Semester 2/ADM (H9ADM)/Project/Dataset/exoTest.csv", stringsAsFactors = F)
new1 = rbind(m,n)

st <- new1
st$LABEL = as.factor(st$LABEL)

###################################
# Plotting FLUX variables         #
# To check for linear correlation #
###################################

par(mfrow=c(1,2))
plot(new1$FLUX.1,new1$FLUX.2, xlab="FLUX 1", ylab = "FLUX 2", pch = "o", col = "red")
plot(new1$FLUX.112,new1$FLUX.132, xlab="FLUX 112", ylab = "FLUX 132", pch = "o", col = "blue")
par(mfrow=c(1,1))

#########################################
# Feature Selection using Random Forest #
#########################################

# Searching noise in dataset which do not influence the target variable
noise <- which(unlist(lapply(st, function(x)length(unique(x))))==1)
noise

# Fit random forest predicting Exoplanets from all predictors
randomf <- randomForest(LABEL~.,data=st, ntree=500)
varImpPlotData <- varImpPlot(randomf)
barplot(summarize_tree(randomf)$importance)
abline (v=300, h =.05)
rfcount <- sum(randomf$importance>.05)

# Selecting top predictors only
imp.features <- c("LABEL", names(summarize_tree(randomf)$importance[1:rfcount]))
st <- st[,imp.features]
str(st)

########################################
# PCA with Scree Plot and Density Matrix
########################################

# Bartlett's Test of Sphericity
bart_spher(st[,-1], use="everything")

# Kaiser Meyer Olkin Statistic
KMOS(st[,-1], use="everything")

# Running PCA
pca_1 = subset(st,select = -c(LABEL))
pca = prcomp( pca_1, center = T, scale. = T)

summ <- summary(pca)
summ_df <- as.data.frame(summ$importance)
summ_df <- t(summ_df)
summ_df <- data.frame(summ_df)
summ_df <- rownames_to_column(as.data.frame(summ_df), var="PrinComp")

#number of PCA with eigenvalues greater than 1
pca_count = sum(summ_df$Standard.deviation > 1)
pca_count

#proportion of variance explained by components
var.pca <- pca$sdev ^ 2
x.var.pca <- var.pca / sum(var.pca)
cum.var.pca <- cumsum(x.var.pca)
plot(cum.var.pca[1:25],xlab="No. of principal components",
     ylab="Cumulative Proportion of variance explained", ylim=c(0,1), type='b')

#screeplot showing bar and line plot
fviz_eig(pca, ncp = 25, xlab="Principal Component")

#choosing the PCA
pca.df <- as.data.frame(pca$x[,1:(pca_count)])
pca.df <- data.frame(LABEL = st$LABEL, pca.df)

#plotting density matrix
par(mfrow=c(2,2))
plot(density(pca.df[pca.df$LABEL == 1,2]), main = "PCA 1, Non Exoplanet")
plot(density(pca.df[pca.df$LABEL == 2,2]), main = "PCA 1, Exoplanet")
plot(density(pca.df[pca.df$LABEL == 1,16]), main = "PCA 15, Non Exoplanet")
plot(density(pca.df[pca.df$LABEL == 2,16]), main = "PCA 15, Exoplanet")
par(mfrow=c(1,1))

#######################################
# Class Balancing & Data Partitioning #
#######################################
set.seed(2018)

index = createDataPartition(pca.df$LABEL, p = .80,  list = FALSE)
train=pca.df[index,]
test=pca.df[-index,]
train$LABEL <- as.numeric(train$LABEL)
test$LABEL <- as.numeric(test$LABEL)
balanced <- ROSE(LABEL ~ ., data = train, N = 5657, p = .33)$data
table(balanced$LABEL)
table(test$LABEL)

#build decision tree models
tree <- rpart(LABEL~ ., data = balanced)
#make predictions on data
pred <- predict(tree, newdata = test)
#accuracy
roc.curve(test$LABEL, pred)

# forming testing and training datasets by retaining required PCAs
pca_testset <- data.frame(test[,1:pca_count])
pca_trainset <- data.frame(balanced[,1:pca_count])

# binary classification 
pca_testset$LABEL <- as.numeric(pca_testset[,"LABEL"] == "2")
pca_trainset$LABEL <- as.numeric(pca_trainset[,"LABEL"] == "2")

table(pca_trainset$LABEL)
str(pca_testset$LABEL)
any(pca_testset$LABEL == 1) #checking to see if exoplanet exist in test data


###########################
# Preprocessing Completed #
###########################


###########################
# Building XGBoost Algo   #
###########################

set.seed(2018)

# Converting training set into DMatrix for XGBoost
train_matrix <- xgb.DMatrix(as.matrix(pca_trainset %>% select(-LABEL)))
train_labels <- pca_trainset$LABEL
test_matrix <- xgb.DMatrix(as.matrix(pca_testset %>% select(-LABEL)))
test_labels <- pca_testset$LABEL

#Cross Validation and NFolds
xgb_trainctl = trainControl(
  method = "cv",
  number = 5,
  allowParallel = TRUE,
  verboseIter = TRUE,
  returnData = FALSE
)
# Hyperparameter Grid
xgbGrid <- expand.grid(nrounds = c(100, 500),  # n_estimator
                       max_depth = 50,
                       colsample_bytree = seq(0.5, 0.9, length.out = 5),
                       ## The values below are default values from sklearn.
                       eta = c(0.01,0.3),
                       gamma=c(0,1)
                       
)
#set.seed(2018)
#Model Training
xgbModel <- train(train_matrix,train_labels,
                  trControl = xgb_trainctl,
                  tuneGrid = xgbGrid,
                  method = "xgbTree"
                  
)
# Best Hyper Parameter Value
xgbModel$bestTune

#Model Evaluation
predicted <-  predict(xgbModel, test_matrix)
# xgb.save(predicted, "XGBootsModel")  # Save Model and Load on requirement
# predicted <- xgb.load("XGBoostModel") #Load the Model

residuals <-  test_labels - predicted
RMSE <-  sqrt(mean(residuals^2))
cat('The RMS error for the test data is', round(RMSE,3),'\n')
test_mean <-  mean(test_labels)
# Calculate total sum of squares
tssq <-   sum((test_labels - test_mean)^2 )
# Calculate residual sum of squares
rss <-   sum(residuals^2)
# Calculate R-squared
rsq  <-   1 - (rss/tssq)
cat('The R-square of the test data is', round(rsq,3), '\n')


prediction <- as.numeric(predicted > 0.5)
print(head(prediction))
err <- mean(as.numeric(pred > 0.5) != test$LABEL)
print(paste("test-error=", err))

options(repr.plot.width=8, repr.plot.height=4)
my_data = as.data.frame(cbind(predicted = predicted,
                              observed = test_labels))

## Visualisation ##
# Plot predictions vs test data
ggplot(my_data,aes(predicted, observed)) + geom_point(color = "red", alpha = 0.65) +
  geom_smooth(method=lm)+ ggtitle('Linear Regression ') + ggtitle("XGBoosting: Prediction vs Test Data") +
  xlab("Predecited Output ") + ylab("Observed Output") +
  theme(plot.title = element_text(color="darkgreen",size=16,hjust = 0.5),
        axis.text.y = element_text(size=12), axis.text.x = element_text(size=12,hjust=.5),
        axis.title.x = element_text(size=14), axis.title.y = element_text(size=14))


# Use ROCR package to plot ROC Curve
xgb.pred <- prediction(predicted, test_labels)
xgb.perf <- performance(xgb.pred, "tpr", "fpr")

plot(xgb.perf,
     avg="threshold",
     colorize=TRUE,
     lwd=1,
     main="ROC Curve Thresholds",
     print.cutoffs.at=seq(0, 1, by=0.05),
     text.adj=c(-0.5, 0.5),
     text.cex=0.5)
grid(col="lightgray")
axis(1, at=seq(0, 1, by=0.1))
axis(2, at=seq(0, 1, by=0.1))
abline(v=c(0.1, 0.3, 0.5, 0.7, 0.9), col="lightgray", lty="dotted")
abline(h=c(0.1, 0.3, 0.5, 0.7, 0.9), col="lightgray", lty="dotted")
lines(x=c(0, 1), y=c(0, 1), col="black", lty="dotted")

# Set our cutoff threshold
pred.resp <- ifelse(predicted >= 1, 1, 0)
summary(pred.resp)

# Create the confusion matrix
confusionMatrix(as.factor(pred.resp), as.factor(test_labels), positive="1")

#####################################################################################################################################################################

#################################
# Building H2O for deeplearning #
#################################


pca_trainset$LABEL = factor(pca_trainset$LABEL,levels = c(0,1),labels = c(1,2))
pca_testset$LABEL = factor(pca_testset$LABEL,levels = c(0,1),labels = c(1,2))

###########################
#H2o initialization       #
###########################
h2o.init(ip = "localhost",port=54321,max_mem_size = "2650m")
y = "LABEL"
x = setdiff(colnames(pca_trainset),y)


###############################################
#deep learning without hypermetric tuning     #
###############################################

model1 <- h2o.deeplearning(
  model_id = "dl_1", 
  training_frame = as.h2o(pca_trainset), 
  validation_frame = as.h2o(pca_testset),
  x = x,
  y = "LABEL",
  hidden = c(32, 32, 32), # smaller network, runs faster
  epochs = 100, # hopefully converges earlier...
  stopping_metric = "misclassification",
  stopping_tolerance = 0.01,distribution = "multinomial"
  ,nfolds = 10)
summary(model1)
pred2 <- h2o.predict(model1, as.h2o(pca_testset[,-1]))
predictt = pred2$predict

predictt = as.data.frame(predictt)

##################################################
#manual accuracy check without hypermetric tuning#
##################################################

acc=1-mean(pca_testset$LABEL!=predictt$predict)

acc
#######################################################
# confusion matrix of H2owithout hypermetric tuning   #
#######################################################

pca_testset$LABEL=as.factor(pca_testset$LABEL)
confusionMatrix(predictt$predict,pca_testset$LABEL,positive ="2")

###############################################
#deep learning with hypermetric tuning        #
###############################################

###############################################
# tuning of hyperparameters                   #
###############################################


hidden_opt <- list(rep(150,3), c(200, 150, 75, 50), 100)
l1_opt <- c(1e-5,1e-7)#
activation <- c("Tanh", "RectifierWithDropout", "Maxout")
hyper_params <- list(hidden = hidden_opt,
                     l1 = l1_opt, activation = activation)



# performs the grid search
grid_id <- "dl_grid"
model_dl_grid <- h2o.grid(
  algorithm = "deeplearning", # name of the algorithm 
  grid_id = grid_id, 
  training_frame = as.h2o(pca_trainset),
  validation_frame =as.h2o( pca_testset), 
  x = x, 
  y = "LABEL",
  epochs = 10,
  stopping_metric = "misclassification",
  stopping_tolerance = 1e-2, # stop when logloss does not improve by >=1% for 2 scoring events
  stopping_rounds = 2,
  score_validation_samples = 10000, # downsample validation set for faster scoring
  hyper_params = hyper_params  
)

# RectifierWithDropout [200, 150, 75, 50] 1.0E-7 ADM_ann_grid_model_13 0.007376736840404132
# find the best model and evaluate its performance
#stopping_metric <- 'accuracy'
stopping_metric <- 'accuracy'

###############################################
#getting best value model                     #
###############################################

sorted_models <- h2o.getGrid(
  grid_id = grid_id, 
  sort_by = stopping_metric,
  decreasing = TRUE
)
sorted_models
best_model <- h2o.getModel(sorted_models@model_ids[[1]])
class(best_model)


class(best_model)
pred11 <- h2o.predict(best_model, as.h2o(pca_testset[,-1]))
predictt11 = pred11$predict

predictt11 = as.data.frame(predictt11)

##################################################
#manual accuracy check without hypermetric tuning#
##################################################

acc=1-mean(pca_testset$LABEL!=predictt11$predict)

acc
#######################################################
# confusion matrix of H2O wit hypermetric tuning   #
#######################################################
library(MLmetrics)
pca_testset$LABEL=as.factor(pca_testset$LABEL)
confusionMatrix(predictt11$predict,pca_testset$LABEL,positive ="2")
summary(F1_Score(pca_testset$LABEL,predictt11$predict))
F1_Score(pca_testset$LABEL,predictt$predict)


#####################################################################################################################################################################

###################################
# Convolutional Neural Network    #
###################################

c = length(pca_testset$LABEL)

x_test = pca_testset[1:c,2:15]
y_test = pca_testset[,1]
x_test=as.matrix(x_test)
x_test = array_reshape(x_test[1:c,], c(dim(x_test[1:c,]), 1))
y_test = array(y_test,dim=c)
y_test = to_categorical(y_test[1:c], 2)
y_test

#make pca_trainset
b = length(pca_trainset$LABEL)

x_train = pca_trainset[1:b,2:15]
y_train = pca_trainset[,1]

#y_train= t(y_train)

#y_train =(y_train)-1
x_train = as.matrix(x_train)
str(x_train)
x_train = array_reshape(x_train[1:b,], c(dim(x_train[1:b,]), 1))

y_train = array(y_train,dim=b)
y_train = to_categorical(y_train[1:b], 2)

k=ncol(x_train)
k


# Running CNN 1D model
model <- keras_model_sequential() 
model %>% 
  layer_conv_1d(filters=10, kernel_size=10,  activation = "relu",  input_shape=c(k, 1)) %>%
  #layer_conv_1d(filters=5, kernel_size=10,  activation = "relu")%>%
  #layer_conv_1d(filters=12, kernel_size=10,  activation = "relu")%>%
  #layer_global_max_pooling_1d() %>%
  #layer_conv_1d(filters=4, kernel_size=10,  activation = "relu")
  layer_max_pooling_1d(pool_size = 4) %>%
  layer_flatten() %>% 
  layer_dense(units = 10, activation = 'relu') %>%
  layer_dense(units = 2, activation = 'softmax')
summary(model)

summary(model)
model %>% compile(
  loss = 'binary_crossentropy',
  optimizer = optimizer_rmsprop(),
  metrics = c('accuracy')
)
history <- model %>% fit(
  x_train, y_train, 
  epochs = 60, batch_size = 30, 
  validation_split = 0.2
)
model %>% evaluate(x_test, y_test)
class(x_test)
pred1 =model%>%predict_classes(x_test)
class(pred1)
table(predicted=pred1,actual=pca_testset[,1])

confusionMatrix(as.factor(pred1),as.factor(pca_testset[,1]))
F1_Score(pred1,pca_testset[,1])



################################################################################################################################################################################################################################################

