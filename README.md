# Predicting Loan Eligibility Readme
The goal of our project is to create a machine learning model that's able to predict loan eligibility based on a set of features from a data set we found on Kaggle 

# Loading Libraries 

```python
# Data Analysis
import pandas as pd
import numpy as np

# Data Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# EDA
from dataprep.eda import create_report

# Interactivity
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"
from IPython.display import display

# Preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer

# Model Evaluation
from sklearn.model_selection import train_test_split, KFold
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import learning_curve
from sklearn.model_selection import validation_curve
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import roc_curve, auc
from lazypredict.Supervised import LazyClassifier, LazyRegressor

# Classifiers
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier

# Explainability:
import shap

# Saving & Loading Best Model
import pickle

# GUI
#import gradio as gr
import tkinter as tk
```
# Loading Data
```python
df = pd.read_csv("train_split.csv")
df.head()
```
# Exploratory Data Analysis

**Analysis of Null Values**
```python
# Count null values for each feature
null_counts = df.isnull().sum()

# Sort the null counts in descending order
sorted_nulls = null_counts.sort_values(ascending=False)

# Select the top 15 features with highest null values
top_10_nulls = sorted_nulls.head(15)

# Create a bar plot
fig, ax = plt.subplots(figsize=(12, 8))  # Increase figure size
top_10_nulls.plot(kind='bar', ax=ax)

# Set plot title and axis labels
ax.set_title('Top 10 Features with Highest Null Values', fontsize=16)  # Increase font size
ax.set_xlabel('Feature', fontsize=14)  # Increase font size
ax.set_ylabel('Number of Null Values', fontsize=14)  # Increase font size

# Rotate x-axis labels for better visibility
plt.xticks(rotation=45, ha='right', fontsize=12)  # Increase font size and adjust horizontal alignment

# Display the plot
plt.tight_layout()  # Adjust layout to prevent labels from being cut off
plt.show()

```

<img width="530" alt="Screenshot 2024-04-24 163402" src="https://github.com/MarcusH25/I320D_final_project/assets/123523085/fd7fce9a-d872-451e-807f-411f430a3f95">


# Correlation Matrix
``` python
numerical_df = df.select_dtypes(include=[np.number])  # Ensure to include import numpy as np if not already done

# Creating the heatmap
plt.figure(figsize=(18, 15))
heatmap = sns.heatmap(numerical_df.corr(), annot=True, cmap='coolwarm', linewidths=0.5, linecolor='white', fmt=".2f", annot_kws={"size": 6})

# Rotate the x-axis labels
plt.xticks(rotation=90)

# Set the font size for x and y tick labels
plt.xticks(fontsize=8)
plt.yticks(fontsize=8)

# Adjust the spacing between subplots
plt.subplots_adjust(left=0.3, right=0.95, top=0.95, bottom=0.3)

# Add a title
plt.title('Correlation Matrix', fontsize=14)

# Show the plot
plt.show()
```
<img width="463" alt="Screenshot 2024-04-24 164506" src="https://github.com/MarcusH25/I320D_final_project/assets/123523085/f19b447e-a7f5-4a08-abaf-a3373fe201d2">


# Data prep 
``` python

# Assuming df is your dataframe
# Perform EDA
report = create_report(df)


# Assuming report is your EDA report generated using dataprep
display(report)

```
# Explanation of Dropping Specific Features
#Explanation of Dropping Specific Features

##### **High Null**
*   **Verification_status_joint** (63971)
*   **desc** (54849)
*   **mths_since_last_record** (54349)
*   **mths_since_last_major_delinq** (32831)
*   **mths_since_last_major_derog** (48155)
*   **batch_enrolled** (10264)
*   **emp_title** (3826)
*   **tot_coll_amt** (5124)
*   **tot_cur_bal**  (5124)
*   **total_rev_hi_lim** (5124)

#####**Missing Vlaues:**

*   **verification_status_joint** (100% missing)

*   **mths_since_last_record** (83% missing)

##### **Does not provide significant:**
*   **zip_code**
*   **title**
*   **addr_state**
*   **member_id**
*   **sub_grade**
#### **Highly correlated features:**

*   **Loan_amnt**
*   **funded_amnt**
*   **List itemfunded_amnt_inv**


# Dropping columns with too many nulls and duplicates

```python
columns_to_drop = ['mths_since_last_delinq', 'mths_since_last_record', 'mths_since_last_major_derog', 'pymnt_plan', 'desc', 'verification_status_joint', 'tot_coll_amt', 'collections_12_mths_ex_med', 'batch_enrolled', 'zip_code', 'title', 'addr_state', 'member_id', 'funded_amnt', 'funded_amnt_inv', 'total_rec_late_fee', 'recoveries', 'collection_recovery_fee', 'acc_now_delinq', 'delinq_2yrs', 'pub_rec', 'application_type', 'grade', 'sub_grade', 'total_rev_hi_lim', 'total_acc',"emp_title"]
df = df.drop(columns=columns_to_drop)
```


# **Preprocessing Features**

**Dropping the Target Variable to Prevent Impact from Scaling or Encoding**
```python
target_variable = df['loan_status']
df = df.drop(columns=['loan_status'])
```

```python
df['last_week_pay'] = df['last_week_pay'].astype(str)
df['last_week_pay'].replace("[^0-9]","",regex=True,inplace=True)
df['last_week_pay'].replace("","-1",regex=True,inplace=True)
df['last_week_pay'] = df['last_week_pay'].apply(lambda x: x.strip())
df.last_week_pay = df.last_week_pay.astype(int)
```

```python
# Extract the number form emp_length to make it numeric
df['emp_length'] = df['emp_length'].astype(str)
df['emp_length'].replace("[^0-9]","",regex=True,inplace=True)
df['emp_length'].replace("","-1",regex=True,inplace=True)
df['emp_length'] = df['emp_length'].apply(lambda x: x.strip())
df.emp_length = df.emp_length.astype(int)

# Use -1 stands for unknown
df['emp_length'].fillna(value='-1',inplace=True)
df['emp_length']

```

```python
# Remove "months" from the variable term to make it numeric
df.term = df.term.apply(lambda x: x.split(' ')[0])
df.term = df.term.astype(int)
df['term']
```

# Implementing standard scaling on numerical features

```python
scaler = StandardScaler()

# Identify the numerical columns
numerical_columns = ['loan_amnt', 'int_rate', 'annual_inc', 'dti', 'inq_last_6mths', 'open_acc', 'revol_bal', 'revol_util', 'total_rec_int', 'tot_cur_bal','last_week_pay']

# Fit the scaler on the entire dataset and transform the numerical columns
scaler = StandardScaler()
df[numerical_columns] = scaler.fit_transform(df[numerical_columns])

```

# Encoding categorical features

```python
categorical_columns = df.select_dtypes(include=['object']).columns

# Create a ColumnTransformer object with OneHotEncoder
preprocessor = ColumnTransformer(
    transformers=[('encoder', OneHotEncoder(handle_unknown='ignore'), categorical_columns)],
    remainder='passthrough'
)

# Fit the preprocessor on the training data and transform both training and testing data
df = preprocessor.fit_transform(df)

df = pd.DataFrame(df, columns=preprocessor.get_feature_names_out())
```

**Concatenating the Target Variable Back into the Data Frame**
```python
df = pd.concat([df, target_variable], axis=1)
```

# Splitting Data into 80/20

```python
X = df.drop('loan_status', axis=1)
y = df['loan_status']

# Split the data into train and test sets with 80/20 ratio
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Check shapes
print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("y_train shape:", y_train.shape)
print("y_test shape:", y_test.shape)
```

# Lazy predict

```python
# LazyClassifier for classification tasks
clf = LazyClassifier(predictions=True)
models, predictions = clf.fit(X_train, X_test, y_train, y_test)
print(models)
```

<img width="436" alt="image" src="https://github.com/MarcusH25/I320D_final_project/assets/123523085/7c694eb3-0e95-4439-9949-b41e06d2572e">


# Classification

**Fitting eXtreme Gradient Boosting**

```python
# Initialize the XGBoost classifier with regularization parameters
xgb_clf = XGBClassifier(reg_alpha=0.5, reg_lambda=1.0, gamma=1.0)

# Extract feature names from the training data
feature_names = X_train.columns.tolist()

# Fit the XGBoost classifier on the training data
xgb_clf.fit(X_train, y_train)

# Perform 5-fold cross-validation on the training data
cv_scores = cross_val_score(xgb_clf, X_train, y_train, cv=5, scoring='accuracy')

# Calculate the mean accuracy
mean_cv_accuracy = cv_scores.mean()
print(f"Mean cross-validation accuracy: {mean_cv_accuracy * 100:.2f}%")

# Make predictions on the test data
y_pred = xgb_clf.predict(X_test)

# Print the classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
```
<img width="334" alt="image" src="https://github.com/MarcusH25/I320D_final_project/assets/123523085/6c8081f6-c585-41d4-b1ab-5fd023e0d038">

**Fitting Light Gradient-Boosting Machine**
```python
# Initialize the LGBMClassifier
lgbm_clf = LGBMClassifier()

# Fit the LGBMClassifier on the training data
lgbm_clf.fit(X_train, y_train)

# Perform 5-fold cross-validation on the training data
cv_scores = cross_val_score(lgbm_clf, X_train, y_train, cv=5, scoring='accuracy')

# Calculate the mean accuracy
mean_cv_accuracy = cv_scores.mean()
print(f"Mean cross-validation accuracy (LGBMClassifier): {mean_cv_accuracy * 100:.2f}%")

# Make predictions on the test data
y_pred = lgbm_clf.predict(X_test)

# Print the classification report
print("\nClassification Report (LGBMClassifier):")
print(classification_report(y_test, y_pred))
```
<img width="331" alt="image" src="https://github.com/MarcusH25/I320D_final_project/assets/123523085/75021447-5ecd-4309-8301-088485a5cc8a">


**Fitting RandomForest**

```python
# Initialize the imputer (Here, using the mean strategy as an example)
imputer = SimpleImputer(strategy='mean')

# Initialize the RandomForest classifier
rf_clf = RandomForestClassifier(n_estimators=100, max_depth=10, min_samples_split=10, random_state=42)

# Assume X_train, X_test, and y_train are already defined and contain NaN values

# Impute missing values in X_train and X_test
X_train_imputed = imputer.fit_transform(X_train)
X_test_imputed = imputer.transform(X_test)

# Extract feature names from the training data after imputation
feature_names = X_train.columns.tolist()

# Fit the RandomForest classifier on the imputed training data
rf_clf.fit(X_train_imputed, y_train)

# Perform 5-fold cross-validation on the imputed training data
cv_scores = cross_val_score(rf_clf, X_train_imputed, y_train, cv=5, scoring='accuracy')

# Calculate the mean accuracy
mean_cv_accuracy = cv_scores.mean()
print(f"Mean cross-validation accuracy: {mean_cv_accuracy * 100:.2f}%")

# Make predictions on the imputed test data
y_pred = rf_clf.predict(X_test_imputed)

# Print the classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
```
<img width="319" alt="image" src="https://github.com/MarcusH25/I320D_final_project/assets/123523085/61aace81-f7b5-43a8-a9db-1581293c184a">


# Model Evaluation 

**Learning Curve**

```python
# Assuming you have already fit the XGBClassifier model: xgb_clf.fit(X_train, y_train)
# And you have split your data into X_train, X_test, y_train, y_test

# Calculate the learning curve
train_sizes, train_scores, test_scores = learning_curve(
    estimator=xgb_clf,
    X=X_train,
    y=y_train,
    train_sizes=np.linspace(0.1, 1.0, 10),
    cv=5,
    scoring='accuracy',
    n_jobs=-1
)

# Calculate the mean and standard deviation for train and test scores
train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)
test_mean = np.mean(test_scores, axis=1)
test_std = np.std(test_scores, axis=1)

# Plot the learning curve
plt.figure(figsize=(10, 6))
plt.plot(train_sizes, train_mean, color='blue', label='Training accuracy')
plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1, color='blue')

plt.plot(train_sizes, test_mean, color='green', label='Cross-validation accuracy')
plt.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, alpha=0.1, color='green')

plt.xlabel('Training set size')
plt.ylabel('Accuracy')
plt.title('Learning Curves')
plt.legend(loc='best')
plt.show()
```

photo goes here!

**Validation Curve**

```python
# Assuming you have already fit the XGBClassifier model: xgb_clf.fit(X_train, y_train)
# And you have split your data into X_train, X_test, y_train, y_test

# Define the hyperparameter to investigate
param_name = 'max_depth'
param_range = np.arange(2, 21)

# Calculate the validation curve
train_scores, test_scores = validation_curve(
    estimator=xgb_clf,
    X=X_train,
    y=y_train,
    param_name=param_name,
    param_range=param_range,
    scoring='accuracy',
    cv=5,
    n_jobs=-1
)

# Calculate the mean and standard deviation for train and test scores
train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)
test_mean = np.mean(test_scores, axis=1)
test_std = np.std(test_scores, axis=1)

# Plot the validation curve
plt.figure(figsize=(10, 6))
plt.plot(param_range, train_mean, color='blue', label='Training accuracy')
plt.fill_between(param_range, train_mean - train_std, train_mean + train_std, alpha=0.1, color='blue')

plt.plot(param_range, test_mean, color='green', label='Cross-validation accuracy')
plt.fill_between(param_range, test_mean - test_std, test_mean + test_std, alpha=0.1, color='green')

plt.xlabel(f'{param_name}')
plt.ylabel('Accuracy')
plt.title('Validation Curve')
plt.legend(loc='best')
plt.show()
```

photo goes here!

**Ablation Study: "leave-one-feature-out"**


