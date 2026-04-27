#-----------------------------------------------------------------------------------------#
# STAT606 Practical Assignment
# Predict whether machine downtime requires maintenance
#-----------------------------------------------------------------------------------------#

# Load libraries
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
# Setup parameters
# Chose using AUC instead of F1 because:
# downtime datasets are often imbalanced &
# AUC evaluates ranking ability across thresholds
# AUC = TPR(FPR)

seed = 606
train_frac <- 0.7
metric <- "AUC"
folds <- 5

#-------------------------------------------------------------------------------------------#
# Load dataset


library(readr)
iportal_downtimes_emea <- read_csv("C:/Users/mambaza/Desktop/UKZN PGDP Data Science/Study stuff/Semester 1/STAT606WA1  Applied Binary Classification and Matching/Group Assignment/Assignment files/iportal downtimes emea.csv")
View(iportal_downtimes_emea)

df <- data.frame(iportal_downtimes_emea)

summary(df)
str(df)

#------------------------------------------------------------------------------------------#
# Data cleaning

# Create target variable

# Since the dataset does not explicitly tell us whether maintenance was required,
# we create a proxy target:
# Longer downtime events + technical failures are more likely to require maintenance

# Create duration threshold
duration_threshold <- median(
  df$DowntimeDuration,
  na.rm = TRUE
)

# Create maintenance target
df$MaintenanceRequired <- ifelse(
  df$DowntimeDuration > duration_threshold &
    grepl(
      "technical|mechanical|electrical|failure|breakdown|fault",
      df$DowntimeType,
      ignore.case = TRUE
    ),
  1,
  0
)

# Convert target to factor
df$MaintenanceRequired <- factor(df$MaintenanceRequired)

# Check class balance
table(df$MaintenanceRequired)
prop.table(table(df$MaintenanceRequired))

# Remove duplicates
# Motivation: Duplicate downtime logs distort model learning.
df <- unique(df)

# Handle missing numerical values
# Why median?
# Less sensitive to extreme downtime values.

df$FaultDuration[is.na(df$FaultDuration)] <- median(
  df$FaultDuration,
  na.rm = TRUE
)

# Handle missing categorical values
df$Area[is.na(df$Area)] <- "Unknown"
df$Department[is.na(df$Department)] <- "Unknown"

# Remove high-cardinality identifiers
# IDs provide no predictive value.

df <- df %>%
  select(-DowntimeId, -MessageId)

# IMPORTANT:
# Remove leakage columns because target was created using Reason
# MessageText may also directly reveal maintenance keywords

df <- df %>%
  select(-Reason, -MessageText)

#-----------------------------------------------------------------------------------------#
# Basic sample characteristics

dim(df)
summary(df)
table(df$MaintenanceRequired)

#-----------------------------------------------------------------------------------------#
# Using existing attributes to create new ones to be used in the models
# we think they would be more suitable/informative (Feature engineering)

# Create hour feature
df$DowntimeStartDatetime <- as.POSIXct(
  df$DowntimeStartDatetime
)

df$Hour <- as.numeric(
  format(df$DowntimeStartDatetime, "%H")
)

# Create day of week feature
df$DayOfWeek <- weekdays(
  df$DowntimeStartDatetime
)

# Emergency keyword feature
df$EmergencyFlag <- ifelse(
  grepl(
    "emergency|critical|urgent",
    df$DowntimeDescription,
    ignore.case = TRUE
  ),
  1,
  0
)

#-----------------------------------------------------------------------------------------#
# Convert categorical variables to factors

df <- df %>%
  mutate(across(where(is.character), as.factor))

# Specify dataframe + target

target <- "MaintenanceRequired"

#----------------------------------------------------------------------------------------#
# Train/Test split (70/30)

set.seed(seed)

idx <- createDataPartition(df[[target]], p = train_frac, list = FALSE)

training_set <- df[idx, ]
test_set <- df[-idx, ]

# sanity checks

# sanity check (good practice, also helps marks)
table(training_set[[target]])
table(test_set[[target]])

#----------------------------------------------------------------------------------------#
# Double check the downtime data (columns) for any hms

sapply(training_set, class)
sapply(test_set, class)

# Drop problematic datetime/time columns since H2O cannot parse hms
# We also have several raw datetime columns that may not add much value
# and can create unnecessary complexity

training_set <- training_set %>%
  select(
    -TimestampTime,
    -LastChangeTime,
    -DowntimeStartDatetime,
    -DowntimeEndDatetime,
    -Timestamp,
    -ProductionDay,
    -TimestampDate
  )

test_set <- test_set %>%
  select(
    -TimestampTime,
    -LastChangeTime,
    -DowntimeStartDatetime,
    -DowntimeEndDatetime,
    -Timestamp,
    -ProductionDay,
    -TimestampDate
  )

# Fix FaultDuration = factor likely be numeric
# Why?
# Fault duration is a continuous measurement, and keeping it as a factor may hurt:
# Decision tree
# converting it

training_set$FaultDuration <- as.numeric(
  as.character(training_set$FaultDuration)
)

test_set$FaultDuration <- as.numeric(
  as.character(test_set$FaultDuration)
)

#---------------------------------------------------------------------------------------#
# Recalculate predictors AFTER removing columns
# This avoids H2O column mismatch errors

predictors <- setdiff(
  names(training_set),
  target
)

print(predictors)

#---------------------------------------------------------------------------------------#
# Initialize H2O

h2o.init()

train_h2o <- as.h2o(training_set)
test_h2o <- as.h2o(test_set)

#---------------------------------------------------------------------------------------#
# Fit Naive Bayes Model

nb <- h2o.naiveBayes(
  x = predictors,
  y = target,
  training_frame = train_h2o,
  laplace = 0,
  nfolds = folds,
  seed = seed
)

# Training performance
h2o.performance(nb)