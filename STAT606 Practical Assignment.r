#-----------------------------------------------------------------------------------------#
# STAT606 Practical Assignment
# Predict whether machine downtime requires maintenance
#-----------------------------------------------------------------------------------------#

# Load libraries required for data manipulation, modelling and evaluation
library(dplyr)
library(caTools)
library(caret)
library(pROC)
library(h2o)
library(rpart)
library(rpart.plot)
library(readr)

options(scipen = 999)

#-------------------------------------------------------------------------------------------#
# Setup parameters for reproducibility and model evaluation

# We use AUC instead of F1 because:
# - downtime data is often imbalanced
# - AUC evaluates ranking performance across all thresholds
# - It is more stable for business classification problems

seed = 606
train_frac <- 0.7
metric <- "AUC"
folds <- 5
threshold <- 0.5

#-------------------------------------------------------------------------------------------#
# Load dataset from local directory (raw EMEA downtime operational data)

iportal_downtimes_emea <- read_csv("C:/Users/mambaza/Desktop/UKZN PGDP Data Science/Study stuff/Semester 1/STAT606WA1  Applied Binary Classification and Matching/Group Assignment/Assignment files/iportal downtimes emea.csv")

df <- data.frame(iportal_downtimes_emea)

# Initial data understanding to inspect structure, data types and distributions
summary(df)
str(df)

#------------------------------------------------------------------------------------------#
# Data cleaning and creation of target variable (MaintenanceRequired)

# Missing downtime types are replaced with "Unknown" so model can still learn from them
df$DowntimeType[is.na(df$DowntimeType)] <- "Unknown"

# We define a threshold based on the 75th percentile of downtime duration
# This helps us identify unusually long downtime events

duration_threshold <- quantile(df$DowntimeDuration, 0.75, na.rm = TRUE)

# Create binary target variable:
# 1 = maintenance required (long duration or failure-related keyword present)
# 0 = no maintenance required

df$MaintenanceRequired <- ifelse(
  df$DowntimeDuration > duration_threshold |
    grepl("technical|mechanical|electrical|failure|breakdown|fault",
          df$DowntimeType, ignore.case = TRUE),
  1, 0
)

df$MaintenanceRequired <- factor(df$MaintenanceRequired)

# Check class balance to ensure both classes exist
table(df$MaintenanceRequired)
prop.table(table(df$MaintenanceRequired))

# Ensure dataset is valid for classification
if(length(unique(df$MaintenanceRequired)) < 2){
  stop("Target variable has only one class. Check logic.")
}

# Remove duplicate records to prevent bias in model learning
df <- unique(df)

# Handle missing numeric values using median (robust to outliers)
df$FaultDuration[is.na(df$FaultDuration)] <- median(df$FaultDuration, na.rm = TRUE)

# Handle missing categorical values
df$Area[is.na(df$Area)] <- "Unknown"
df$Department[is.na(df$Department)] <- "Unknown"

# Remove identifier columns that do not contribute to prediction
df <- df %>% select(-DowntimeId, -MessageId)

# Remove leakage columns that could directly reveal the target outcome
df <- df %>% select(-Reason, -MessageText)

#-----------------------------------------------------------------------------------------#
# Basic dataset characteristics after cleaning

# We check dataset shape, structure and target distribution
dim(df)
summary(df)
table(df$MaintenanceRequired)

#-----------------------------------------------------------------------------------------#
# Feature engineering to improve model predictive capability

# Convert timestamp into datetime format for feature extraction
df$DowntimeStartDatetime <- as.POSIXct(df$DowntimeStartDatetime)

# Extract hour of downtime (captures operational shift patterns)
df$Hour <- as.numeric(format(df$DowntimeStartDatetime, "%H"))

# Extract day of week (captures operational cycle behaviour)
df$DayOfWeek <- weekdays(df$DowntimeStartDatetime)

# Create emergency indicator based on keywords in downtime description
df$EmergencyFlag <- ifelse(
  grepl("emergency|critical|urgent",
        df$DowntimeDescription, ignore.case = TRUE),
  1, 0
)

# Convert all character variables into factors for modelling compatibility
df <- df %>% mutate(across(where(is.character), as.factor))

target <- "MaintenanceRequired"

#----------------------------------------------------------------------------------------#
# Split dataset into training and testing sets (70% train, 30% test)

set.seed(seed)

idx <- createDataPartition(df[[target]], p = train_frac, list = FALSE)

training_set <- df[idx, ]
test_set <- df[-idx, ]

training_set[[target]] <- as.factor(training_set[[target]])
test_set[[target]] <- as.factor(test_set[[target]])

# Check class balance in both training and test sets
table(training_set[[target]])
table(test_set[[target]])

# Ensure training set contains both classes
if(length(unique(training_set[[target]])) < 2){
  stop("Training set has only one class.")
}

#----------------------------------------------------------------------------------------#
# Remove timestamp columns that are not supported by H2O models

training_set <- training_set %>% select(
  -TimestampTime, -LastChangeTime, -DowntimeStartDatetime,
  -DowntimeEndDatetime, -Timestamp, -ProductionDay, -TimestampDate
)

test_set <- test_set %>% select(
  -TimestampTime, -LastChangeTime, -DowntimeStartDatetime,
  -DowntimeEndDatetime, -Timestamp, -ProductionDay, -TimestampDate
)

# Ensure FaultDuration remains numeric after transformations
training_set$FaultDuration <- as.numeric(as.character(training_set$FaultDuration))
test_set$FaultDuration <- as.numeric(as.character(test_set$FaultDuration))

#---------------------------------------------------------------------------------------#
# Fix missing values, NaNs and infinite values before training H2O models
# This is critical because H2O models cannot handle NaN or Inf values

training_set[sapply(training_set, is.infinite)] <- NA
test_set[sapply(test_set, is.infinite)] <- NA

training_set[] <- lapply(training_set, function(x) ifelse(is.nan(x), NA, x))
test_set[] <- lapply(test_set, function(x) ifelse(is.nan(x), NA, x))

num_cols <- sapply(training_set, is.numeric)

for (col in names(training_set)[num_cols]) {
  med <- median(training_set[[col]], na.rm = TRUE)
  training_set[[col]][is.na(training_set[[col]])] <- med
  test_set[[col]][is.na(test_set[[col]])] <- med
}

cat_cols <- sapply(training_set, is.factor)

for (col in names(training_set)[cat_cols]) {
  training_set[[col]][is.na(training_set[[col]])] <- "Unknown"
  test_set[[col]][is.na(test_set[[col]])] <- "Unknown"
}

#---------------------------------------------------------------------------------------#
# Define predictor variables for modelling

predictors <- setdiff(names(training_set), target)

#---------------------------------------------------------------------------------------#
# Initialize H2O cluster for distributed machine learning

h2o.init()

train_h2o <- as.h2o(training_set)
test_h2o <- as.h2o(test_set)

train_h2o[[target]] <- as.factor(train_h2o[[target]])
test_h2o[[target]] <- as.factor(test_h2o[[target]])

#---------------------------------------------------------------------------------------#
# 1. Decision Tree model (implemented using single-tree Random Forest for stability)

DT <- h2o.randomForest(
  x = predictors,
  y = target,
  training_frame = train_h2o,
  ntrees = 1,
  max_depth = 5,
  min_rows = 10,
  seed = seed
)

preds_DT <- as.data.frame(h2o.predict(DT, test_h2o))

test_DT_pred <- cbind(test_set,
                      pred_prob = as.numeric(preds_DT[,3]))

roc_dt_test <- roc(as.numeric(as.character(test_DT_pred[[target]])),
                   test_DT_pred$pred_prob)

#---------------------------------------------------------------------------------------#
# 2. Logistic Regression model (captures linear relationships in downtime behaviour)

LR <- h2o.glm(
  x = predictors,
  y = target,
  training_frame = train_h2o,
  family = "binomial",
  lambda = 0,
  seed = seed
)

preds_LR <- as.data.frame(h2o.predict(LR, test_h2o))

test_LR_pred <- cbind(test_set,
                      pred_prob = as.numeric(preds_LR[,3]))

roc_LR_test <- roc(as.numeric(as.character(test_LR_pred[[target]])),
                   test_LR_pred$pred_prob)

#---------------------------------------------------------------------------------------#
# 3. Random Forest model (captures complex non-linear relationships)

RF <- h2o.randomForest(
  x = predictors,
  y = target,
  training_frame = train_h2o,
  ntrees = 100,
  max_depth = 20,
  nfolds = folds,
  seed = seed
)

preds_RF <- as.data.frame(h2o.predict(RF, test_h2o))

test_RF_pred <- cbind(test_set,
                      pred_prob = as.numeric(preds_RF[,3]))

roc_RF_test <- roc(as.numeric(as.character(test_RF_pred[[target]])),
                   test_RF_pred$pred_prob)

#---------------------------------------------------------------------------------------#
# 4. Naive Bayes model (assumes conditional independence of predictors)

nb <- h2o.naiveBayes(
  x = predictors,
  y = target,
  training_frame = train_h2o,
  laplace = 1,
  seed = seed
)

preds_nb <- as.data.frame(h2o.predict(nb, test_h2o))

test_nb_pred <- cbind(test_set,
                      pred_prob = as.numeric(preds_nb[,3]))

roc_nb_test <- roc(test_nb_pred[[target]],
                   test_nb_pred$pred_prob)

#---------------------------------------------------------------------------------------#
# Model performance comparison using AUC (best model selection)

model_results <- data.frame(
  Model = c("Decision Tree","Logistic Regression","Random Forest","Naive Bayes"),
  AUC = c(auc(roc_dt_test),
          auc(roc_LR_test),
          auc(roc_RF_test),
          auc(roc_nb_test))
)

print(model_results)

best_model <- model_results$Model[which.max(model_results$AUC)]
print(best_model)

#---------------------------------------------------------------------------------------#
# ROC curve comparison for all models

plot(roc_dt_test, col="red", lwd=2)
lines(roc_LR_test, col="green", lwd=2)
lines(roc_RF_test, col="purple", lwd=2)
lines(roc_nb_test, col="blue", lwd=2)

legend("bottomright",
       legend=c("Decision Tree","Logistic Regression","Random Forest","Naive Bayes"),
       col=c("red","green","purple","blue"),
       lwd=2)

#---------------------------------------------------------------------------------------#
# Feature importance based on best performing model

if(best_model == "Random Forest") {
  
  var_imp <- h2o.varimp(RF)
  print(head(var_imp, 5))
  
} else if(best_model == "Decision Tree") {
  
  var_imp <- h2o.varimp(DT)
  print(head(var_imp, 5))
  
} else if(best_model == "Logistic Regression") {
  
  var_imp <- h2o.varimp(LR)
  print(head(var_imp, 5))
  
} else {
  
  print("Naive Bayes does not provide stable feature importance")
}

#---------------------------------------------------------------------------------------#
# Shutdown H2O cluster after modelling

h2o.shutdown(prompt = FALSE)