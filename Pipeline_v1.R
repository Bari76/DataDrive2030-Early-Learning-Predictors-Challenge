###Clear the environment####
rm(list = ls())

###Load the packages####
library(tidyverse)
library(scales)
library(caret)
library(data.table)

#Load the dataset####
path = "C:/Users/kamau/Desktop/Google_Data_Analytics_cert/EarlyChildHoodEdu/Data"

train = read.csv(paste0(path,"/Train.csv"),na.strings = c(" ","NA"))
glimpse(train)
as_tibble(train)
test = read.csv(paste0(path,"/Test.csv"),na.strings = c(" ","NA"))
glimpse(test)
as_tibble(test)
sub_sample = read.csv(paste0(path,"/SampleSubmission.csv"))
glimpse(sub_sample)

#Combine the two data sets####

df = rbind(within(train,rm('child_id','target')),within(test,rm('child_id')))
glimpse(df)

#Missing Values####
na.cols = which(colSums(is.na(df))>0)
sort(colSums(sapply(df[na.cols],is.na)),decreasing = T) %>% View()
paste('There are', length(na.cols), 'columns with missing values')
na.cols = as.data.frame(na.cols)
na.cols$name = rownames(na.cols)
na.cols$name
# A Dataframe of all the cols with NAs####
df_1 = df[,names(df) %in% na.cols$name]
miss = sapply(df,function(x) percent(sum(is.na(x))/nrow(df)))
miss_percent = data.frame(variables = names(miss),percent_missing=miss) %>% 
  as.tibble() %>% 
  filter(percent_missing>"70%") %>% 
  arrange(desc(percent_missing))

#Variables to be imputed
var_impute=df[,!names(df) %in% miss_percent$variables]
#Variables to be dropped
var_drop = df[,names(df) %in% miss_percent$variables]

##Drop variables with more than 70% missing values
df[,names(var_drop)]  =NULL
glimpse(df)
sum(is.na(df))
colSums(is.na(df))

##Impute the missing values####
#Get the mode
my_mode <- function (x, na.rm) {
  xtab <- table(x)
  xmode <- names(which(xtab == max(xtab)))
  if (length(xmode) > 1) xmode <- ">1 mode"
  return(xmode)
}

for (var in 1:ncol(df)) {
  if (class(df[,var])=="numeric") {
    df[is.na(df[,var]),var] <- mean(df[,var], na.rm = TRUE)
  } else if (class(df[,var]) %in% c("character", "factor")) {
    df[is.na(df[,var]),var] <- my_mode(df[,var], na.rm = TRUE)
  }
}

####See the NAs ####
na.cols_1 = which(colSums(is.na(df))>0)
#The  cols with Nas are all pri_time
#Needs more investigation.
sort(colSums(sapply(df[na.cols_1],is.na)),decreasing = T) 
paste('There are', length(na.cols_1), 'columns with missing values')
select(df,contains(c("pri_ti"))) %>% View()
#Impute them with the mode.
for(i in which(sapply(df,is.numeric))){
  df[is.na(df[, i]),i] <- median(df[, i], na.rm = T)
}

####Convert variables into their correct format####
df = df %>% 
  mutate_at(vars(matches(c("date","dob"))),function(x)return(as.Date(x)))


#Dividing the df into DATE, NUM & CATEGORICAL####
num_features = names(which(sapply(df,is.numeric)))
cat_features = names(which(sapply(df,is.character)))
date_col = select(df,contains(c("date","dob"))) %>% glimpse()

#Correlation####
df.numeric = df[num_features]
corr.df = cbind(df.numeric[1:8585,],train['target'])
correlation = cor(corr.df)
#cols that show strong correlation with target
corr.target = as.matrix(sort(correlation[,'target'],decreasing = T))
corr.idx = names(which(apply(corr.target,1,function(x) (x>0.5 | x < -.5))))
corrplot::corrplot(as.matrix(correlation[corr.idx,corr.idx]),
                   type = 'upper',method = 'color',addCoef.col = 'black',
                   tl.cex = .7,cl.cex = .7,number.cex = .7)

####Correlation-Use another function####
library(corrr)
res.cor <- correlate(df.numeric)
res.cor

######Median ELOM Score per province####
train[,c('prov_best','target')] %>%
  group_by(prov_best) %>%
  summarise(median.ELOM = round(median(target, na.rm = TRUE),3)) %>% 
  arrange(median.ELOM) %>%
  mutate(provice.sorted = factor(prov_best, levels=prov_best))   %>%
  ggplot(aes(x=provice.sorted, y=median.ELOM)) +
  geom_point() +
  geom_text(aes(label = median.ELOM, angle = 45), vjust = 2) +
  theme_minimal() +
  labs(x='Provice', y='Median ELOM') +
  theme(text = element_text(size=12),
        axis.text.x = element_text(angle=45))

#PCA####
require(factoextra)
pmatrix = df.numeric %>%
  prcomp( center = TRUE, scale. = TRUE)

pcaVar <- as.data.frame(c(get_pca_var(pmatrix)))

# lets
pcaVarNew <- pcaVar[, 1:10]

#######Pre-Processing####
require(psych)
library(moments)
# linear models assume normality from dependant variables 
# transform any skewed data into normal
skewed <- apply(df.numeric, 2, skewness)
skewed <- skewed[(skewed > 0.8) | (skewed < -0.8)]

kurtosis <- apply(df.numeric, 2, kurtosi)
kurtosis <- kurtosis[(kurtosis > 3.0) | (kurtosis < -3.0)]
# normalize the data
library(caret)
scaler <- preProcess(df.numeric)
df.numeric <- predict(scaler, df.numeric)


#For the rest of the categoric features we can 
#one-hot encode each value to get as many splits in 
#the data as possible

# one hot encoding for categorical data####
# sparse data performs better for trees/xgboost
library(fastDummies)
dummy <- dummyVars(" ~ .",data=df[,cat_features])

df.categoric = df.categoric %>% select(-contains(c("date","dob")))
df.categoric <- data.frame(predict(dummy,newdata=df[,cat_features]))


## Extract date features from the date columns####
for (col in date_col){
  df[paste(col, 'Year', sep=" ")] = format(df[col], "%Y")
  df[paste(col, 'Month', sep=" ")] = format(df[col], "%m")
  df[paste(col, 'Day', sep=" ")] = format(df[col], "%d")
}

###Combine####
df <- cbind(df.numeric, df.categoric,date_col)

###NearZero####
nzv.data <- nearZeroVar(df, saveMetrics = TRUE)

# take any of the near-zero-variance perdictors
drop.cols <- rownames(nzv.data)[nzv.data$nzv == TRUE]

df <- df[,!names(df) %in% drop.cols]

paste('The dataframe now has', dim(df)[1], 'rows and', dim(df)[2], 'columns')

#####Modelling####
#XGBoost
library(xgboost)

#Splitting data(HandOut Method)####
y_train <- log(train$target+1)
x_train <- df[1:8585,]

x_test <- df[8585:nrow(df),]
#For multi-core training because we are training alot of trees
library(doSNOW) 
cl=makeCluster(6,type = "SOCK")
registerDoSNOW(cl)

# Set seed for reproducability
set.seed(1234)
dtrain <- xgb.DMatrix(data.matrix(x_train), label = y_train)
dtest <- xgb.DMatrix(data.matrix(x_test))

#Cross-validation####
cv.ctrl <- trainControl(method = "repeatedcv", repeats = 1,number = 4, 
                        allowParallel=T)
#Tuning the parameters####
xgb.grid <- expand.grid(nrounds = 750,
                        eta = c(0.01,0.005,0.001),
                        max_depth = c(4,6,8),
                        colsample_bytree=c(0,1,10),
                        min_child_weight = 2,
                        subsample=c(0,0.2,0.4,0.6),
                        gamma=0.01)

#Train the model####
xgb_tune <- train(data.matrix(x_train),
                  y_train,
                  method="xgbTree",
                  trControl=cv.ctrl,
                  tuneGrid=xgb.grid,
                  verbose=T,
                  metric="RMSE", nthread =3)

stopCluster(cl)

###Tuning Parameters
xgb_params <- list(
  booster = 'gbtree',
  objective = 'reg:linear',
  colsample_bytree=1,
  eta=0.005,
  max_depth=4,
  min_child_weight=3,
  alpha=0.3,
  lambda=0.4,
  gamma=0.01, # less overfit
  subsample=0.6,
  seed=5,
  silent=TRUE)
#How well my model generalizes to testing data
xgb.cv(xgb_params, dtrain, nrounds = 5000, nfold = 4,
       early_stopping_rounds = 500)
bst <- xgb.train(xgb_params,dtrain, nrounds = 1000)#,early_stopping_rounds = 300, watchlist = list(train=dtrain))))

#Evaluation metric####
rmse_eval <- function(y_train, y.pred) {
  mse_eval <- sum((y_train - exp(y.pred)-1)^2) / length(y_train)
  return(sqrt(mse_eval))
}

y_pred.xgb <- predict(bst, dtrain)
rmse_eval(y_train, y_pred.xgb)

#Note: our predictions are in logarithmic form since 
#we made a transformation due to the skewness in the
#target variable so we will need to tranform our predictions
#back into their original form by taking f(x)=ex−1
y_pred.xgb <- as.double(predict(bst, dtest))
y_pred.xgb <- as.double(exp(y_pred.xgb) - 1)

# bst had a root mean square error of 45.169

###Feature importance####
model.names <- dimnames(dtrain)[[2]]

importance_matrix <- xgb.importance(model.names, model = bst)

xgb.plot.importance(importance_matrix[1:15])

### I haven't tried it yet.
##Creating submission
library(SHAPforxgboost)

# Step 1: Select some observations
X <- data.matrix(df[sample(nrow(df), 1000), 1:20])
# Step 2: Crunch SHAP values
shap_value <- shap.prep(fit_xgb, X_train = X)
# Step 3: SHAP importance
shap.plot.summary(shap)

# Initialize the dictionary
f <- list(
  f1 = vector(),
  f2 = vector(),
  f3 = vector(),
  f4 = vector(),
  f5 = vector(),
  f6 = vector(),
  f7 = vector(),
  f8 = vector(),
  f9 = vector(),
  f10 = vector(),
  f11 = vector(),
  f12 = vector(),
  f13 = vector(),
  f14 = vector(),
  f15 = vector()
)

# Iterate over each shap_value
for (shap_value in shap_values) {
  # Get the indices of the top 15 features
  arr <- order(shap_value, decreasing = TRUE)[1:15]
  
  # Append the features to the respective f* list
  for (ind in 1:15) {
    name_f <- paste0("f", ind)
    f[[name_f]] <- c(f[[name_f]], features[arr[ind]])
  }
}

# Create the SampleSubmission dataframe with the features
SampleSubmission <- data.frame(
  target = preds_test,
  feature_1 = f$f1,
  feature_2 = f$f2,
  feature_3 = f$f3,
  feature_4 = f$f4,
  feature_5 = f$f5,
  feature_6 = f$f6,
  feature_7 = f$f7,
  feature_8 = f$f8,
  feature_9 = f$f9,
  feature_10 = f$f10,
  feature_11 = f$f11,
  feature_12 = f$f12,
  feature_13 = f$f13,
  feature_14 = f$f14,
  feature_15 = f$f15
)

# Create a csv file and upload to zindi 
write.csv(SampleSubmission, 'Baseline.csv', row.names=FALSE)


# ridge, lasso, elasticnet

#library(glmnet)
#library(Matrix)

#Lets train our model using our chosen λ’s

#glm.ridge <- glmnet(x = as.matrix(x_train), y = y_train, alpha = 0, lambda = penalty.ridge )
#glm.lasso <- glmnet(x = as.matrix(x_train), y = y_train, alpha = 1, lambda = penalty.lasso)
#glm.net <- glmnet(x = as.matrix(x_train), y = y_train, alpha = 0.001, lambda = penalty.net)


#Evaluating how well the models performed
#y_pred.ridge <- as.numeric(predict(glm.ridge, as.matrix(x_train)))
#y_pred.lasso <- as.numeric(predict(glm.lasso, as.matrix(x_train)))
#y_pred.net <- as.numeric(predict(glm.net, as.matrix(x_train)))
#rmse_eval(y.true, y_pred.ridge)
#rmse_eval(y.true, y_pred.lasso)
#rmse_eval(y.true, y_pred.net)

##Making Predictions
#y_pred.ridge <- as.double(predict(glm.ridge, as.matrix(x_test)))
#y_pred.lasso <- as.double(predict(glm.lasso, as.matrix(x_test)))
#y_pred.net <- as.double(predict(glm.net, as.matrix(x_test)))

##Converting from log transformation
#y_pred.ridge <- as.double(exp(y_pred.ridge) - 1)
#y_pred.lasso <- as.double(exp(y_pred.lasso) - 1)
#y_pred.net <- as.double(exp(y_pred.net) - 1)

#y_pred.xgb <- as.double(predict(bst, dtest))
#y_pred.xgb <- as.double(exp(y_pred.xgb) - 1)

# take the average of our predictions for our ensemble
#y_pred <- (y_pred.xgb + y_pred.ridge + y_pred.lasso + y_pre.net)/4.0
#head(y_pred)