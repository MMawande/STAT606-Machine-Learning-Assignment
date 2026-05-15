#-----------------------------------------------------------------------------------------#
# STAT606 Practical Assignment
# Predict whether machine downtime requires maintenance
#-----------------------------------------------------------------------------------------#

# Load libraries required for data manipulation, modelling and evaluation
library(dplyr) # for data manipulation and preprocessing
library(caTools) # for splitting into train and test sets
library(caret) # used for performance metric functions
library(pROC) # used for obtaining AUC
library(h2o)
library(readr)

# For plotting Decision Trees
library(rpart)
library(rpart.plot)

# This turns scientific notation off so output is easier to interpret
options(scipen = 999)

#-------------------------------------------------------------------------------------------#
# Setup parameters for reproducibility and model evaluation

# We use AUC instead of F1 because:
# - downtime data is often imbalanced
# - AUC evaluates ranking performance across all thresholds
# - It is more stable for business classification problems

seed <- 606
train_frac <- 0.7
metric <- "AUC"
folds <- 5
threshold <- 0.5

#-------------------------------------------------------------------------------------------#
# Load dataset from local directory (raw EMEA downtime operational data)

iportal_downtimes_emea <- read_csv(
  "C:/Users/mambaza/Desktop/UKZN/PGDM - Data Science/Semester 1/STAT606 - Applied Binary Classification and Matching/R Practicals/My practice scripts/STAT606-Machine-Learning-Assignment/iportal downtimes emea.csv"
)

View(iportal_downtimes_emea)

df <- data.frame(iportal_downtimes_emea)

# Initial data understanding to inspect structure, data types and distributions
df$TimestampTime <- as.character(df$TimestampTime)

summary(df)
str(df)

#------------------------------------------------------------------------------------------#
# Data cleaning and creation of target variable (MaintenanceRequired)

# Missing downtime types are replaced with "Unknown"
# so the model can still learn from them

df$DowntimeType[is.na(df$DowntimeType)] <- "Unknown"

# We define a threshold based on the 75th percentile of downtime duration
# This helps identify unusually long downtime events

duration_threshold <- quantile(
  df$DowntimeDuration,
  0.75,
  na.rm = TRUE
)

# Create binary target variable:
# 1 = maintenance required
# 0 = no maintenance required

df$MaintenanceRequired <- ifelse(
  df$DowntimeDuration > duration_threshold |
    grepl(
      "technical|mechanical|electrical|failure|breakdown|fault",
      df$DowntimeType,
      ignore.case = TRUE
    ),
  1,
  0
)

# Convert target variable to factor with explicit binary labels
# This prevents H2O from internally recoding classes incorrectly

df$MaintenanceRequired <- factor(
  as.character(df$MaintenanceRequired),
  levels = c("0", "1")
)

# Verify class levels
levels(df$MaintenanceRequired)

# Check class balance to ensure both classes exist

table(df$MaintenanceRequired)
prop.table(table(df$MaintenanceRequired))

# Ensure dataset is valid for classification

if(length(unique(df$MaintenanceRequired)) < 2){
  stop("Target variable has only one class. Check logic.")
}

# Remove duplicate records to prevent bias in model learning

df <- unique(df)

# Remove leakage variable used in target construction

df <- df %>% select(-DowntimeType)

#------------------------------------------------------------------------------------------#
# Convert FaultDuration to numeric before handling missing values

df$FaultDuration <- as.numeric(as.character(df$FaultDuration))

# Handle missing numeric values using median (robust to outliers)

df$FaultDuration[is.na(df$FaultDuration)] <- median(
  df$FaultDuration,
  na.rm = TRUE
)

# Handle missing categorical values

df$Area[is.na(df$Area)] <- "Unknown"
df$Department[is.na(df$Department)] <- "Unknown"

# Remove identifier columns that do not contribute to prediction

df <- df %>% select(
  -DowntimeId,
  -MessageId,
  -Reason,
  -MessageText
)

#------------------------------------------------------------------------------------------#
# Feature engineering to improve model predictive capability

# Convert timestamp into datetime format for feature extraction

df$DowntimeStartDatetime <- as.POSIXct(df$DowntimeStartDatetime)

# Extract hour of downtime (captures operational shift patterns)

df$Hour <- as.numeric(format(df$DowntimeStartDatetime, "%H"))

# Extract day of week (captures operational cycle behaviour)

df$DayOfWeek <- weekdays(df$DowntimeStartDatetime)

# Create emergency indicator based on keywords in downtime description

df$EmergencyFlag <- ifelse(
  grepl(
    "emergency|critical|urgent",
    df$DowntimeDescription,
    ignore.case = TRUE
  ),
  1,
  0
)

# Convert all character variables into factors for modelling compatibility

df <- df %>%
  mutate(across(where(is.character), as.factor))


# Define target variable
target <- "MaintenanceRequired"

#----------------------------------------------------------------------------------------#
#----------------------------------------------------------------------------------------#
# Split dataset into training and testing sets (70% train, 30% test)
#
# sample.split() preserves class distribution between train and test sets,

set.seed(seed)

split <- sample.split(
  df[[target]],
  SplitRatio = train_frac
)

training_set <- subset(df, split == TRUE)
test_set <- subset(df, split == FALSE)

# Ensure target variable remains properly coded

training_set[[target]] <- factor(
  training_set[[target]],
  levels = c("0", "1")
)

test_set[[target]] <- factor(
  test_set[[target]],
  levels = c("0", "1")
)

# Check class balance in both training and test sets

table(training_set[[target]])
table(test_set[[target]])

# Ensure training set contains both classes

if(length(unique(training_set[[target]])) < 2){
  stop("Training set has only one class.")
}

#----------------------------------------------------------------------------------------#
# Remove timestamp columns not supported by H2O models

training_set <- training_set %>% select(
  -TimestampTime,
  -LastChangeTime,
  -DowntimeStartDatetime,
  -DowntimeEndDatetime,
  -Timestamp,
  -ProductionDay,
  -TimestampDate
)

test_set <- test_set %>% select(
  -TimestampTime,
  -LastChangeTime,
  -DowntimeStartDatetime,
  -DowntimeEndDatetime,
  -Timestamp,
  -ProductionDay,
  -TimestampDate
)

# Ensure FaultDuration remains numeric after transformations

training_set$FaultDuration <- as.numeric(
  as.character(training_set$FaultDuration)
)

test_set$FaultDuration <- as.numeric(
  as.character(test_set$FaultDuration)
)

#---------------------------------------------------------------------------------------#
# Fix missing values, NaNs and infinite values before training H2O models

training_set[sapply(training_set, is.infinite)] <- NA
test_set[sapply(test_set, is.infinite)] <- NA

training_set[] <- lapply(
  training_set,
  function(x) ifelse(is.nan(x), NA, x)
)

test_set[] <- lapply(
  test_set,
  function(x) ifelse(is.nan(x), NA, x)
)

#---------------------------------------------------------------------------------------#
# Fix missing values before training H2O models

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

# Define predictor variables for modelling

predictors <- setdiff(names(training_set), target)

#---------------------------------------------------------------------------------------#
# Initialize H2O cluster for distributed machine learning

h2o.init()

train_h2o <- as.h2o(training_set)
test_h2o <- as.h2o(test_set)

# Ensure binary response variable remains correctly coded inside H2O

train_h2o[[target]] <- as.factor(train_h2o[[target]])
test_h2o[[target]] <- as.factor(test_h2o[[target]])

# Confirm class labels recognised by H2O

print(h2o.levels(train_h2o[[target]]))
print(h2o.levels(test_h2o[[target]]))

# Numeric response for ROC analysis

y_test_numeric <- as.numeric(as.character(test_set[[target]]))

#---------------------------------------------------------------------------------------#
# 1. Decision Tree model
# Implemented using Random Forest with a small number of trees for stability

DT <- h2o.randomForest(
  x = predictors,
  y = target,
  training_frame = train_h2o,
  ntrees = 8,
  max_depth = 5,
  min_rows = 10,
  seed = seed
)

#---------------------------------------------------------------------------------------#
# 1.2 Decision Tree model using rpart
# Used for visual interpretation of tree structure

set.seed(seed)

DT_rpart <- rpart(
  as.formula(paste(target, "~ .")),
  data = training_set,
  method = "class",
  xval = folds,
  control = rpart.control(
    # cp = 0.03,
    # minsplit = 20,
    maxdepth = 4
  )
)

# Display fitted tree information
DT_rpart

# Visualize the tree

rpart.plot(
  DT_rpart,
  yesno = 1,
  type = 2,
  fallen.leaves = FALSE
)

#---------------------------------------------------------------------------------------#
# Generate predictions from Decision Tree model
#
# H2O returns:
# predict = predicted class
# p0 = probability of class "0"
# p1 = probability of class "1"

preds_DT <- as.data.frame(h2o.predict(DT, test_h2o))

# Use probability of positive class ("1")

prob_col_DT <- "p1"

# Convert probabilities into binary predictions

preds_DT$predicted_class <- ifelse(
  preds_DT$p1 >= threshold,
  1,
  0
)

# View prediction results

View(preds_DT)

# ROC curve based on probabilities

roc_dt_test <- roc(
  y_test_numeric,
  as.numeric(preds_DT[[prob_col_DT]])
)

#---------------------------------------------------------------------------------------#
# 2. Logistic Regression model
# Captures linear relationships in downtime behaviour

LR <- h2o.glm(
  x = predictors,
  y = target,
  training_frame = train_h2o,
  family = "binomial",
  lambda = 0,
  seed = seed
)

# Generate predictions

preds_LR <- as.data.frame(h2o.predict(LR, test_h2o))

# Use probability of positive class ("1")

prob_col_LR <- "p1"

# Convert probabilities into binary predictions

preds_LR$predicted_class <- ifelse(
  preds_LR$p1 >= threshold,
  1,
  0
)

# ROC curve based on probabilities

roc_LR_test <- roc(
  y_test_numeric,
  as.numeric(preds_LR[[prob_col_LR]])
)

#---------------------------------------------------------------------------------------#
# 3. Naive Bayes model
# Assumes conditional independence between predictors

nb <- h2o.naiveBayes(
  x = predictors,
  y = target,
  training_frame = train_h2o,
  laplace = 1,
  seed = seed
)

# Generate predictions

preds_nb <- as.data.frame(h2o.predict(nb, test_h2o))

# Use probability of positive class ("1")

prob_col_nb <- "p1"

# Convert probabilities into binary predictions

preds_nb$predicted_class <- ifelse(
  preds_nb$p1 >= threshold,
  1,
  0
)

# ROC curve based on probabilities

roc_nb_test <- roc(
  y_test_numeric,
  as.numeric(preds_nb[[prob_col_nb]])
)

#---------------------------------------------------------------------------------------#
# Model performance comparison using AUC

model_results <- data.frame(
  Model = c(
    "Decision Tree",
    "Logistic Regression",
    "Naive Bayes"
  ),
  AUC = c(
    auc(roc_dt_test),
    auc(roc_LR_test),
    auc(roc_nb_test)
  )
)

print(model_results)

# Select best model based on highest AUC

best_model <- model_results$Model[
  which.max(model_results$AUC)
]

print(best_model)

#---------------------------------------------------------------------------------------#
# ROC curve comparison for all models

plot(roc_dt_test, col = "red", lwd = 2)

lines(roc_LR_test, col = "green", lwd = 2)

lines(roc_nb_test, col = "blue", lwd = 2)

legend(
  "bottomright",
  legend = c(
    "Decision Tree",
    "Logistic Regression",
    "Naive Bayes"
  ),
  col = c("red", "green", "blue"),
  lwd = 2
)

#---------------------------------------------------------------------------------------#
# Feature importance based on best performing model

if(best_model == "Decision Tree") {
  
  var_imp <- h2o.varimp(DT)
  print(head(var_imp, 5))
  
} else if(best_model == "Logistic Regression") {
  
  var_imp <- h2o.varimp(LR)
  print(head(var_imp, 5))
  
} else if(best_model == "Naive Bayes") {
  
  print("Naive Bayes does not provide stable feature importance")
  
}

#---------------------------------------------------------------------------------------#
# Shutdown H2O cluster after modelling

h2o.shutdown(prompt = FALSE)