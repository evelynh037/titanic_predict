# app.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import numpy as np
import os

LOG_PATH = "/app/output_log_python.txt" 
if os.path.exists(LOG_PATH):
    os.remove(LOG_PATH)

        
        
# Load training and test data
train = pd.read_csv("src/data/train.csv")
test = pd.read_csv("src/data/test.csv")
print('Read in train and test data')

# Feature selection
train = train[["Survived", "Pclass", "Sex", "Age", "Fare", "Embarked"]]
print('Keep "Survived", "Pclass", "Sex", "Age", "Fare", "Embarked" as predictors')

# Preprocessing
# Convert Sex to binary
train["Sex"] = train["Sex"].map({"male": 0, "female": 1})
print('Convert Sex into numerical binary: male(0) and female(1)')

# Fill missing Embarked values with most frequent category
most_common_embarked = train["Embarked"].mode()[0]
train["Embarked"] = train["Embarked"].fillna(most_common_embarked)
print(f'Fill missing Embarked values with most frequent category {most_common_embarked}')

# Fill missing Age values grouped by Pclass and Sex
train['Age'] = train.groupby(['Pclass','Sex'])['Age'].transform(lambda x: x.fillna(x.median()))
print("Fill missing Age values grouped by Pclass and Sex")

# One-hot encode Embarked
train = pd.get_dummies(train, columns=["Embarked"], drop_first=True)
print('One-hot encode Embarked')

# Standardize Age and log-transform Fare
train["Age"] = (train["Age"] - train["Age"].mean()) / train["Age"].std()
train["Fare"] = np.log1p(train["Fare"])
print('Standardize Age and log-transform Fare')

# Train and fit logistic regression
X = train.drop("Survived", axis=1)
y = train["Survived"]
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)
print('Train and fit logistic regression')

# Validate
y_pred = model.predict(X_val)
accuracy = accuracy_score(y_val, y_pred)
print(f"Validation Accuracy: {accuracy:.4f}")

# Prepare test data with the same procedure as test data
test = test[["Pclass", "Sex", "Age", "Fare", "Embarked"]]
test["Sex"] = test["Sex"].map({"male": 0, "female": 1})
test["Embarked"] = test["Embarked"].fillna(most_common_embarked)
test['Age'] = test.groupby(['Pclass','Sex'])['Age'].transform(lambda x: x.fillna(x.median()))
test = pd.get_dummies(test, columns=["Embarked"], drop_first=True)
test["Age"] = (test["Age"] - test["Age"].mean()) / test["Age"].std()
test["Fare"] = test["Fare"].fillna(test["Fare"].median())
test["Fare"] = np.log1p(test["Fare"])

print("Prepare test data with the same procedures performed on the test data, fill empty fare cell with medium fare")

# Predict on test data
test_pred = model.predict(test)
print("Make prediction on test set...")

# dump the predictions into the txt file
with open(LOG_PATH, "a") as f:
    f.write("Test Predictions:\n")
    for pred in test_pred:
        f.write(f"{pred}\n")

print(f"Predictions saved to output_log_python.txt")