#Predicting Loan Eligibility Readme
# The goal of our project is to create a machine learning model that's able to predict loan eligibility
# based on a set of features from a data set we found on Kaggle.


#%% 
#Loading libraries & Data 
# Data Analysis
import pandas as pd
import numpy as np

# Data Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# EDA
#from dataprep.eda import create_report

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
from sklearn.utils import resample

# Model Evaluation
from sklearn.model_selection import train_test_split, KFold
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import learning_curve
from sklearn.model_selection import validation_curve
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import roc_curve, auc

# from lazypredict.Supervised import LazyClassifier, LazyRegressor

# Classifiers
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier

# Explainability:
import shap

# Saving & Loading Best Model
import pickle

# GUI
import tkinter as tk
#%%
# Loading Data 
df = pd.read_csv("train_split.csv")
df.head()

# Getting the df shape 
df.shape
# %% Exploratory Data Analysis

# EDA Graphs 

# Counts of loans 
sns.countplot(x='loan_status', data=df)
plt.title('Count of Loan Status')
plt.xlabel('Loan Status')
plt.ylabel('Count')
plt.show()


#Top 15 Features with Highest Null Values

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
ax.set_title('Top 15 Features with Highest Null Values', fontsize=16)  # Increase font size
ax.set_xlabel('Feature', fontsize=14)  # Increase font size
ax.set_ylabel('Number of Null Values', fontsize=14)  # Increase font size

# Rotate x-axis labels for better visibility
plt.xticks(rotation=45, ha='right', fontsize=12)  # Increase font size and adjust horizontal alignment

# Display the plot
plt.tight_layout()  # Adjust layout to prevent labels from being cut off
plt.show()



# Correaltation maxtrix 


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
#%% Pandas EDA
# Getting data types 
df.info()
df.isnull().sum()
#%%
# Data prep EDA
 
report = create_report(df)

# Assuming report is your EDA report generated using dataprep
display(report) 
#%%
# **Dropping columns with too many nulls and duplicates**
columns_to_drop = ['mths_since_last_delinq', 'mths_since_last_record', 'mths_since_last_major_derog', 'pymnt_plan', 'desc', 'verification_status_joint', 'tot_coll_amt', 'collections_12_mths_ex_med', 'batch_enrolled', 'zip_code', 'title', 'addr_state', 'member_id', 'funded_amnt', 'funded_amnt_inv', 'total_rec_late_fee', 'recoveries', 'collection_recovery_fee', 'acc_now_delinq', 'delinq_2yrs', 'pub_rec', 'application_type', 'grade', 'sub_grade', 'total_rev_hi_lim', 'total_acc',"emp_title"]
df = df.drop(columns=columns_to_drop)
#%%
df.shape
#%% 
# Preprocessing Features



# **Undersampling majority class**



# Separate the majority and minority classes
df_majority = df[df['loan_status'] == 0]
df_minority = df[df['loan_status'] == 1]

# Get the minimum number of samples between the two classes
min_samples = min(len(df_majority), len(df_minority))

# Downsample the majority class without replacement
df_majority_downsampled = resample(df_majority,
                                   replace=False,
                                   n_samples=min_samples,
                                   random_state=42)

# Downsample the minority class without replacement
df_minority_downsampled = resample(df_minority,
                                   replace=False,
                                   n_samples=min_samples,
                                   random_state=42)

# Combine the downsampled majority and minority classes
df_balanced = pd.concat([df_majority_downsampled, df_minority_downsampled])

# Verify the class balance
print(df_balanced['loan_status'].value_counts())# Verify the class balance

# df_majority = df[df['loan_status'] == 0]
# df_minority = df[df['loan_status'] == 1]

# # Downsample the majority class to 50,000 rows
# df_majority_downsampled = resample(df_majority, replace=False, n_samples=15197, random_state=42)

# # Downsample the minority class to 50,000 rows
# df_minority_downsampled = resample(df_minority, replace=False, n_samples=15197, random_state=42)

# # Combine the downsampled majority and minority classes
# df_balanced = pd.concat([df_majority_downsampled, df_minority_downsampled])
# df = df_balanced

# Verify the class balance
print(df['loan_status'].value_counts())

# Dropping target variable to avoid complications 
target_variable = df['loan_status']
df = df.drop(columns=['loan_status'])

# **Data handling of last_week_pay**
df['last_week_pay'] = df['last_week_pay'].astype(str)
df['last_week_pay'].replace("[^0-9]","",regex=True,inplace=True)
df['last_week_pay'].replace("","-1",regex=True,inplace=True)
df['last_week_pay'] = df['last_week_pay'].apply(lambda x: x.strip())
df.last_week_pay = df.last_week_pay.astype(int)

# **Extract the number form emp_length to make it numeric**
df['emp_length'] = df['emp_length'].astype(str)
df['emp_length'].replace("[^0-9]","",regex=True,inplace=True)
df['emp_length'].replace("","-1",regex=True,inplace=True)
df['emp_length'] = df['emp_length'].apply(lambda x: x.strip())
df.emp_length = df.emp_length.astype(int)

# Why do we do this ?

# Use -1 stands for unknown
df['emp_length'].fillna(value='-1',inplace=True)
df['emp_length']

# **Remove "months" from the variable term to make it numeric**
df.term = df.term.apply(lambda x: x.split(' ')[0])
df.term = df.term.astype(int)
df['term']
#%% 
# Implementing standard scaling on numerical features

# Creating an instance of Standardscaler
scaler = StandardScaler()

# Identify the numerical columns
numerical_columns = ['loan_amnt', 'int_rate', 'annual_inc', 'dti', 'inq_last_6mths', 'open_acc', 'revol_bal', 'revol_util', 'total_rec_int', 'tot_cur_bal','last_week_pay']

# Fit the scaler on the entire dataset and transform the numerical columns
scaler = StandardScaler()
df[numerical_columns] = scaler.fit_transform(df[numerical_columns])
# %%
# Encoding categorical features

categorical_columns = df.select_dtypes(include=['object']).columns

# Create a ColumnTransformer object with OneHotEncoder
preprocessor = ColumnTransformer(
    transformers=[('encoder', OneHotEncoder(handle_unknown='ignore'), categorical_columns)],
    remainder='passthrough'
)

# Fit the preprocessor on the training data and transform both training and testing data
df = preprocessor.fit_transform(df)

df = pd.DataFrame(df, columns=preprocessor.get_feature_names_out())

#%% 
# concat loan status back & splitting the data into a 80/20 split 
df = pd.concat([df, target_variable], axis=1)
#%%
# Splitting the data into 80/20 

X = df.drop('loan_status', axis=1)
y = df['loan_status']

# Split the data into train and test sets with 80/20 ratio
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Check shapes
print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("y_train shape:", y_train.shape)
print("y_test shape:", y_test.shape)
# %%
# Lazy predict

# LazyClassifier for classification tasks
# clf = LazyClassifier(predictions=True)
# models, predictions = clf.fit(X_train, X_test, y_train, y_test) 
# print(models)

#%%
print("Value counts in y_train:")
print(y_train.value_counts())

# Check the value counts of the target variable in the test set
print("\nValue counts in y_test:")
print(y_test.value_counts())
#%%
# Classification

#%%
# Initialize the XGBoost classifier
xgb_clf = XGBClassifier()

# Extract feature names from the training data
feature_names = X_train.columns.tolist()

# Set the feature names for the classifier
xgb_clf.fit(X_train, y_train)

# Perform 5-fold cross-validation on the training data
cv_scores = cross_val_score(xgb_clf, X_train, y_train, cv=5, scoring='accuracy')

# Calculate the mean accuracy
mean_cv_accuracy = cv_scores.mean()
print(f"Mean cross-validation accuracy: {mean_cv_accuracy * 100:.2f}%")

# Make predictions on the test data
y_pred = xgb_clf.predict(X_test)

# Convert target variables to integers
y_test = y_test.astype(int)
y_pred = y_pred.astype(int)

# Print the classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
# %%
