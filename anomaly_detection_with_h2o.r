
# functions ----------------------------------------------------------


GetROC_AUC = function(probs, true_Y){
  probsSort = sort(probs, decreasing = TRUE, index.return = TRUE)
  val = unlist(probsSort$x)
  idx = unlist(probsSort$ix) 
  
  roc_y = true_Y[idx];
  stack_x = cumsum(roc_y == 0)/sum(roc_y == 0)
  stack_y = cumsum(roc_y == 1)/sum(roc_y == 1)   
  
  auc = sum((stack_x[2:length(roc_y)]-stack_x[1:length(roc_y)-1])*stack_y[2:length(roc_y)])
  return(auc)
}

Binarize_Features <- function(data_set, features_to_ignore=c(), 
                              leave_out_one_level=FALSE, 
                              max_level_count=20) {
  require(dplyr)
  text_features <- c(names(data_set[sapply(data_set, is.character)]), names(data_set[sapply(data_set, is.factor)]))
  for (feature_name in setdiff(text_features, features_to_ignore)) {
    print(paste('Binarizing:', feature_name))
    feature_vector <- as.character(data_set[,feature_name])
    
    # check that data has more than one level
    if (length(unique(feature_vector)) == 1)
      next
    
    # We set any non-data to text
    feature_vector[is.na(feature_vector)] <- 'NA'
    feature_vector[is.infinite(feature_vector)] <- 'INF'
    feature_vector[is.nan(feature_vector)] <- 'NAN'
    
    # only give us the top x most popular categories
    temp_vect <- data.frame(base::table(feature_vector)) %>% dplyr::arrange(desc(Freq)) %>% 
      head(max_level_count)
    feature_vector <- ifelse(feature_vector %in% temp_vect$feature_vector, feature_vector, 'Other')
    
    # loop through each level of a feature and create a new column
    first_level=TRUE
    for (newcol in unique(feature_vector)) {
      if (leave_out_one_level & first_level) {
        # avoid dummy trap and skip first level
        first_level=FALSE
        next
      }
      
      data_set[,paste0(feature_name,"_",newcol)] <- ifelse(feature_vector==newcol,1,0)
    }
    # remove original feature
    data_set <- data_set[,setdiff(names(data_set),feature_name)]
  }
  return (data_set)
}


# you will need Java for your OS 
# Java SE Development Kit 8 Downloads
# install.packages("h2o")
library(h2o)

h2oServer <- h2o.init(nthreads=-1)
homedir <- "/Users/manuel/Documents/h2o/"


# diabetes data  ----------------------------------------------------------

require(RCurl)
binData <- getBinaryURL("https://archive.ics.uci.edu/ml/machine-learning-databases/00296/dataset_diabetes.zip",
                        ssl.verifypeer=FALSE)

conObj <- file("dataset_diabetes.zip", open = "wb")
writeBin(binData, conObj)
# don't forget to close it
close(conObj)

# open diabetes file
files <- unzip("dataset_diabetes.zip")
diabetes <- read.csv(files[1], stringsAsFactors = FALSE)
head(diabetes,1)

# generalize outcome name
outcome_name <- 'readmitted'

# ignore non-modeling variables
features_to_ignore <- c('encounter_id', 'patient_nbr', 'readmitted')

# transform all "?" to 0s
diabetes[diabetes == "?"] <- NA

# remove zero variance - ty James http://stackoverflow.com/questions/8805298/quickly-remove-zero-variance-variables-from-a-data-frame
diabetes <- diabetes[sapply(diabetes, function(x) length(levels(factor(x,exclude=NULL)))>1)]

# prep outcome variable to those readmitted under 30 days
diabetes$readmitted <- ifelse(diabetes$readmitted == "<30",1,0)


# drop large factors
diabetes <- subset(diabetes, select=-c(diag_1, diag_2, diag_3))
dim(diabetes)

print(head(diabetes,1))

# transform type id fields into binary fields
diabetes$admission_type_id <- as.character(diabetes$admission_type_id)
diabetes$discharge_disposition_id <- as.character(diabetes$discharge_disposition_id)
diabetes$admission_source_id <- as.character(diabetes$admission_source_id)


# binarize all text data
diabetes_ready_df <- Binarize_Features(data_set = diabetes,
                                       features_to_ignore = features_to_ignore,
                                       max_level_count = 100)


library(tidyverse)
names(diabetes_ready_df) <- make.names(names(diabetes_ready_df))

# for good measure and to keep things fast and slim- near zero var to limit the data set
library(caret)
dim(diabetes_ready_df)
nzv <- nearZeroVar(diabetes_ready_df, saveMetrics = TRUE)
length(rownames(nzv[nzv$nzv==FALSE,]))
diabetes_ready_df <- diabetes_ready_df[,rownames(nzv[nzv$nzv==FALSE,])]
dim(diabetes_ready_df)

# h2o  ----------------------------------------------------------

# you will need Java for your OS 
# Java SE Development Kit 8 Downloads
# install.packages("h2o")
library(h2o)

h2oServer <- h2o.init(nthreads=-1)
homedir <- "/Users/manuel/Documents/h2o/"

# feed into h2o but remove non-modeling variables first
diabetes_ready_h2o <-as.h2o(dplyr::select(diabetes_ready_df, -encounter_id, -patient_nbr,-readmitted) , destination_frame="diabetes.hex")

# run unsupervised deep learning with autoencoder
feature_names <-  names(dplyr::select(diabetes_ready_df, -encounter_id, -patient_nbr, -readmitted))
anomaly_model <- h2o.deeplearning(x=feature_names,
                                  training_frame=diabetes_ready_h2o,
                                  seed=0000,
                                  hidden=c(20),
                                  epoch=1,
                                  activation="Tanh",
                                  autoencoder=T,
                                  ignore_const_cols=F,
                                  reproducible=T)


reconstruction_error  = as.data.frame(h2o.anomaly(anomaly_model, diabetes_ready_h2o, per_feature=FALSE))
head(reconstruction_error )
range(reconstruction_error )

# get worst offending rows
plot(sort(reconstruction_error$Reconstruction.MSE[sample(length(reconstruction_error$Reconstruction.MSE), 2000)]), 
     main='Reconstruction Error',
     xlab='Patient', ylab='Error')


# attach error score to row and pull highest and lowest for investigation
diabetes_ready_df$Reconstruction_Error <-reconstruction_error$Reconstruction.MSE

# show one full row with reconstruction error
head(diabetes_ready_df, 1)


# sort diabetes dataset by errors
diabetes_ready_df <- dplyr::arrange(diabetes_ready_df, desc(Reconstruction_Error))

# show biggest error
head(diabetes_ready_df, 1)

# shot smallest error
tail(diabetes_ready_df, 1)


# take 10 rows with highest reconstruction errors and compare with 10 rows with the least errors
difference <- c()
for (feature in feature_names) {
  high_errors <- mean(head(diabetes_ready_df[,feature], 10))
  low_errors <- mean(tail(diabetes_ready_df[,feature],10))
  difference <- c(difference, high_errors - low_errors)
}

plot(sort(difference),type="l", col="Blue", pch=2, main='Directional Differences between\n High Error & Low Error Features', ylab='Variable Average Difference',xlab='Feature')
abline(h = 0, col='gray')

feature_divergeance_set <- data.frame('feature'=feature_names, 'avg_difference'=difference)
feature_divergeance_set <- dplyr::arrange(feature_divergeance_set, desc(difference))
head(feature_divergeance_set,5)
tail(feature_divergeance_set,5)



reconstruction_error  = as.data.frame(h2o.anomaly(anomaly_model, diabetes_ready_h2o, per_feature=TRUE))
head(reconstruction_error,1)

print(dim(diabetes_ready_df))
print(dim(reconstruction_error))
 

error_data_all <- c()
for (feature_id in seq(names(reconstruction_error))) {
  error_data <- data.frame(feature=feature_names[feature_id],
                           value=diabetes_ready_df[,feature_names[feature_id]], 
                           patient_id=diabetes_ready_df$patient_nbr,
                           error=reconstruction_error[,feature_id])
  error_data %>% dplyr::arrange(desc(error)) -> error_data
  
  # pick highest 5 patient errors per feature
  error_data <- head(error_data, 5)
  error_data_all <- rbind(error_data_all, error_data)
  
}

# how many patients do have that appear in the list multiple times?
head(sort(table(error_data_all$patient_id), decreasing = TRUE),5)

 
max_patient_id <- as.numeric(names(head(sort(table(error_data_all$patient_id), decreasing = TRUE),1)))

print(dplyr::filter(diabetes_ready_df, patient_nbr==max_patient_id))

# lets model the data with all the data then with varying thresholds of reconstruction errors
# rebuild train_df_auto with best observations


# split data set into training and testing with seed so you can reproduce the split
set.seed(1234)
split <- sample(nrow(diabetes_ready_df), floor(0.5*nrow(diabetes_ready_df)))
diabetes_train_df <- diabetes_ready_df[split,]
diabetes_eval_df <-  diabetes_ready_df[-split,]
 
library(glmnet)
x <- as.matrix(diabetes_train_df[,feature_names])
y <- as.factor(diabetes_train_df[,outcome_name])
set.seed(1234) 
cvfit = cv.glmnet(as(x, "dgCMatrix"),y=y,alpha=0,family='binomial')
coef(cvfit, s = "lambda.min")


x <- as.matrix(diabetes_eval_df[,feature_names])
simple_preds <- predict(cvfit, x)
print(GetROC_AUC(probs = simple_preds, true_Y = diabetes_eval_df[,outcome_name]) )


# improve base score using row splits

# find worst row-based errors and model separately
# #################################################
# feed into h2o but remove non-modeling variables first
diabetes_train_h2o <-as.h2o(dplyr::select(diabetes_train_df, -encounter_id, -patient_nbr,-readmitted) , destination_frame="diabetes.hex")

# run unsupervised deep learning with autoencoder
feature_names <-  names(dplyr::select(diabetes_train_df, -encounter_id, -patient_nbr, -readmitted))
anomaly_model <- h2o.deeplearning(x=feature_names,
                                  training_frame=diabetes_train_h2o,
                                  hidden=c(20),
                                  seed = 0000,
                                  epoch=1,
                                  activation="Tanh",
                                  autoencoder=T,
                                  ignore_const_cols=F,
                                  reproducible=T)


reconstruction_error  = as.data.frame(h2o.anomaly(anomaly_model, diabetes_train_h2o, per_feature=FALSE))
head(reconstruction_error )
range(reconstruction_error )

reconstruction_error  = as.data.frame(h2o.anomaly(anomaly_model, diabetes_ready_h2o, per_feature=FALSE))
head(reconstruction_error)
plot(sort(reconstruction_error$Reconstruction.MSE[sample(nrow(reconstruction_error), 1000)]))

# get 0.5 percent threshold
range(reconstruction_error$Reconstruction.MSE)
model_cutoff <- sort(reconstruction_error$Reconstruction.MSE)[floor(length(reconstruction_error$Reconstruction.MSE) * .995)]
print(model_cutoff)

# gather training portion
dim(diabetes_train_df)
diabetes_train_df_temp <- diabetes_train_df
diabetes_train_df_temp$error_recon <- reconstruction_error$Reconstruction.MSE
diabetes_train_df_temp <- dplyr::filter(diabetes_train_df_temp, 
                                        error_recon < model_cutoff)
dim(diabetes_train_df_temp)

x <- as.matrix(diabetes_train_df_temp[,feature_names])
y = as.factor(diabetes_train_df_temp[,outcome_name])
set.seed(1234) 
cvfit = cv.glmnet(as(x, "dgCMatrix"),y=y,alpha=0,family='binomial')

# gather testing data poriton
# reconstruction_error  = as.data.frame(h2o.anomaly(anomaly_model, diabetes_test_h2o, per_feature=FALSE))
# summary(reconstruction_error)
# plot(sort(reconstruction_error$Reconstruction.MSE[sample(nrow(reconstruction_error), 1000)]))

diabetes_eval_df_temp <- diabetes_eval_df
diabetes_eval_df_temp$error_recon <- reconstruction_error$Reconstruction.MSE
dim(diabetes_eval_df_temp)
diabetes_eval_df_temp <- dplyr::filter(diabetes_eval_df_temp, 
                                       error_recon < model_cutoff)
dim(diabetes_eval_df_temp)

x <- as.matrix(diabetes_eval_df_temp[,feature_names])
preds <- predict(cvfit, x)
preds1 <- preds
outcome1 <- diabetes_eval_df_temp$readmitted


# gather training portion
diabetes_train_df_temp <- diabetes_train_df
diabetes_train_df_temp$error_recon <- reconstruction_error$Reconstruction.MSE
# dim(diabetes_eval_df_temp)
diabetes_train_df_temp <- dplyr::filter(diabetes_train_df_temp, 
                                        error_recon >= model_cutoff)
dim(diabetes_train_df_temp)

x <- as.matrix(diabetes_train_df_temp[,feature_names])
y = as.factor(diabetes_train_df_temp[,outcome_name])
set.seed(1234) 
cvfit = cv.glmnet(as(x, "dgCMatrix"),y=y,alpha=0,family='binomial')

# gather testing data poriton

# gather training portion
diabetes_eval_df_temp <- diabetes_eval_df
diabetes_eval_df_temp$error_recon <- reconstruction_error$Reconstruction.MSE
dim(diabetes_eval_df_temp)
diabetes_eval_df_temp <- dplyr::filter(diabetes_eval_df_temp, 
                                       error_recon >= model_cutoff)
dim(diabetes_eval_df_temp)

x <- as.matrix(diabetes_eval_df_temp[,feature_names])

preds <- predict(cvfit, x)


preds2 <- preds
outcome2 <- diabetes_eval_df_temp$readmitted

allpreds <- rbind(preds1, preds2)
alloutcomes <- c(outcome1, outcome2)

# compare score 
print(GetROC_AUC(probs = preds1, true_Y = outcome1) )

print(GetROC_AUC(probs = preds2, true_Y = outcome2) )

print(GetROC_AUC(probs = allpreds, true_Y = alloutcomes) )

print(GetROC_AUC(probs =  simple_preds , true_Y = diabetes_eval_df[,outcome_name]))

print(GetROC_AUC(probs = c(simple_preds, preds1), true_Y = c(diabetes_eval_df[,outcome_name], outcome1)))
print(GetROC_AUC(probs = c(simple_preds, preds2), true_Y = c(diabetes_eval_df[,outcome_name], outcome2)))


# remove high-error features
reconstruction_error  = as.data.frame(h2o.anomaly(anomaly_model, diabetes_train_h2o, per_feature=TRUE))
head(reconstruction_error )
range(reconstruction_error )
 
error_data_all <- c()
for (feature_id in seq(names(reconstruction_error))) {
  error_data <- data.frame(feature=feature_names[feature_id],
                           error=sum(reconstruction_error[,feature_id]))
 
  error_data_all <- rbind(error_data_all, error_data)
}

error_data_all %>% arrange(desc(error)) -> error_data_all

# top feature to remove
print(head(error_data_all,1))

# get benchmark score using all features
feature_names <-  names(dplyr::select(diabetes_train_df, -encounter_id, -patient_nbr, -readmitted))
x <- as.matrix(diabetes_train_df[,feature_names])
y = as.factor(diabetes_train_df[,outcome_name])
set.seed(1234) 
cvfit = cv.glmnet(as(x, "dgCMatrix"),y=y,alpha=0,family='binomial')
# evaluate
x <- as.matrix(diabetes_eval_df[,feature_names])
preds <- predict(cvfit, x)
print(GetROC_AUC(probs =  preds , true_Y = diabetes_eval_df[,outcome_name]))

features_to_remove <- c('insulin_Up','insulin_Down') # 0.6384215

# get score without most error-prone feature
feature_names <-  names(dplyr::select(diabetes_train_df, -encounter_id, -patient_nbr, -readmitted))
feature_names <- feature_names[!feature_names %in% features_to_remove]
x <- as.matrix(diabetes_train_df[,feature_names])
y = as.factor(diabetes_train_df[,outcome_name])
set.seed(1234) 
cvfit = cv.glmnet(as(x, "dgCMatrix"),y=y,alpha=0,family='binomial')
 
# evaluate
x <- as.matrix(diabetes_eval_df[,feature_names])
preds <- predict(cvfit, x)
print(GetROC_AUC(probs =  preds , true_Y = diabetes_eval_df[,outcome_name]))
