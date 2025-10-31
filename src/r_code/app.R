# app.R
library(readr)
library(dplyr)
library(tibble)

# Clear old log file if exists
file.remove("/app/output_log_r.txt")


# Load training and test data
train <- read_csv("/app/src/data/train.csv", show_col_types = FALSE )
test  <- read_csv("/app/src/data/test.csv", show_col_types = FALSE)
print('[1] "Read in train and test data"')

# Feature selection
train <- train %>% select(Survived, Pclass, Sex, Age, Fare, Embarked)
print('Keep "Survived", "Pclass", "Sex", "Age", "Fare", "Embarked" as predictors')

# Convert Sex to binary
train <- train %>% mutate(Sex = ifelse(Sex == "male", 0, 1))
print('Convert Sex into numerical binary: male(0) and female(1)')

# Fill missing Embarked values with most frequent category
most_common_embarked <- train %>% count(Embarked) %>% arrange(desc(n)) %>% slice(1) %>% pull(Embarked)
train <- train %>% mutate(Embarked = ifelse(is.na(Embarked), most_common_embarked, Embarked))
print(paste('Fill missing Embarked values with most frequent category', most_common_embarked))

# Fill missing Age values grouped by Pclass and Sex
train <- train %>%
  group_by(Pclass, Sex) %>%
  mutate(Age = ifelse(is.na(Age), median(Age, na.rm = TRUE), Age)) %>%
  ungroup()
print("Fill missing Age values grouped by Pclass and Sex")

# One-hot encode Embarked
train <- train %>%
  mutate(Embarked_S = ifelse(Embarked == "S", 1, 0),
         Embarked_Q = ifelse(Embarked == "Q", 1, 0)) %>%
  select(-Embarked)
print("One-hot encode Embarked")

# Standardize Age and log-transform Fare
train <- train %>%
  mutate(Age = scale(Age),
         Fare = log1p(Fare))
print("Standardize Age and log-transform Fare")

# Train logistic regression
X <- train %>% select(-Survived)
y <- train$Survived
train_data <- cbind(y, X)
model <- glm(y ~ ., data = train_data, family = binomial)
print("Train and fit logistic regression")

# Predict on training validation split (manual split)
set.seed(42)
train_index <- sample(1:nrow(train_data), size = 0.8*nrow(train_data))
train_subset <- train_data[train_index, ]
val_subset <- train_data[-train_index, ]

val_pred_prob <- predict(model, val_subset, type = "response")
val_pred <- ifelse(val_pred_prob > 0.5, 1, 0)
accuracy <- mean(val_pred == val_subset$y)
print(sprintf("Validation Accuracy: %.4f", accuracy))

# Prepare test data
test <- test %>% select(Pclass, Sex, Age, Fare, Embarked)
test <- test %>% mutate(Sex = ifelse(Sex == "male", 0, 1))
test <- test %>% mutate(Embarked = ifelse(is.na(Embarked), most_common_embarked, Embarked))
test <- test %>%
  group_by(Pclass, Sex) %>%
  mutate(Age = ifelse(is.na(Age), median(Age, na.rm = TRUE), Age)) %>%
  ungroup()

test <- test %>%
  mutate(Embarked_S = ifelse(Embarked == "S", 1, 0),
         Embarked_Q = ifelse(Embarked == "Q", 1, 0)) %>%
  select(-Embarked)

test <- test %>%
  mutate(Age = scale(Age),
         Fare = ifelse(is.na(Fare), median(Fare, na.rm = TRUE), Fare),
         Fare = log1p(Fare))
print("Prepare test data with the same procedures performed on the test data, fill empty fare cell with medium fare")

# Predict on test data
test_pred_prob <- predict(model, test, type = "response")
test_pred <- ifelse(test_pred_prob > 0.5, 1, 0)
print("Make prediction on test set...")

# Dump the predictions into the txt file
log_file <- file("/app/output_log_r.txt", open = "a")
writeLines("Test Predictions:", log_file)
writeLines(as.character(test_pred), log_file)
close(log_file)

# Test accuracy
correct_answers <- read_csv("/app/src/data/gender_submission.csv", show_col_types = FALSE)
test_accuracy <- mean(test_pred == correct_answers$Survived)
print(sprintf("Test Accuracy: %.4f", test_accuracy))