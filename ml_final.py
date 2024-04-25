

#Predicting Loan Eligibility Readme
# The goal of our project is to create a machine learning model that's able to predict loan eligibility
# based on a set of features from a data set we found on Kaggle.


# %% 
# **Loading libraries & Data** 
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
from imblearn.over_sampling import SMOTE

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
import lazypredict 
from lazypredict.Supervised import LazyClassifier, LazyRegressor
# Explainability:
import shap
import dataprep 
from dataprep.datasets import load_dataset
from dataprep.eda import create_report

# Saving & Loading Best Model
import pickle

# GUI
import tkinter as tk
#%%
# **Loading Data**
df = pd.read_csv("train_split.csv")
df.head()

# Getting the df shape 
df.shape
# %% **Exploratory Data Analysis**

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
#%% **Pandas EDA**
# Getting data types 
df.info()
df.isnull().sum()
#%%
# **Data prep EDA**
 
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
# **Preprocessing Features**

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
# **Creating an instance of Standardscaler**
scaler = StandardScaler()

# Identify the numerical columns
numerical_columns = ['loan_amnt', 'int_rate', 'annual_inc', 'dti', 'inq_last_6mths', 'open_acc', 'revol_bal', 'revol_util', 'total_rec_int', 'tot_cur_bal','last_week_pay']

# Fit the scaler on the entire dataset and transform the numerical columns
scaler = StandardScaler()
df[numerical_columns] = scaler.fit_transform(df[numerical_columns])
# %%
# **Encoding categorical features**

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
# **Concat loan status back & splitting the data into a 80/20 split** 
df = pd.concat([df, target_variable], axis=1)
#%%
# **Splitting the data into 80/20** 

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
# **Lazy predict**

# LazyClassifier for classification tasks
clf = LazyClassifier(predictions=True)
models, predictions = clf.fit(X_train, X_test, y_train, y_test) 
print(models)

 #%%

print("Value counts in y_train:")
print(y_train.value_counts())

# Check the value counts of the target variable in the test set
print("\nValue counts in y_test:")
print(y_test.value_counts())
#%%
# **Classification**

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
# Fitting Light Gradient boosting Machine 

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
# %%

# Fitting RandomForest Classifier 

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
# %% 
# **Model Evaluation**

# Learning Curve Plot

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


# Validation Curve

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
# %% 

# **ROC Cruve & AUC**

y_pred_proba = xgb_clf.predict_proba(X_test)[:, 1]

fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)

roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.show()
# %%
# **Ablation Study "leave-one feature-out"**

# Get the feature names
feature_names = X_train.columns.tolist()

# Initialize a list to store the cross-validation scores for each feature
cv_scores = []

# Iterate over each feature
for feature in feature_names:
    # Create a copy of X_train without the current feature
    X_train_reduced = X_train.drop(feature, axis=1)

    # Perform cross-validation predictions on the reduced training data
    y_pred = cross_val_predict(xgb_clf, X_train_reduced, y_train, cv=5)

    # Calculate the cross-validation accuracy
    score = accuracy_score(y_train, y_pred)

    # Store the cross-validation score
    cv_scores.append(score)

    print(f"Cross-validation accuracy without {feature}: {score:.4f}")

# Find the feature with the highest cross-validation score (most important)
most_important_feature_idx = np.argmax(cv_scores)
most_important_feature = feature_names[most_important_feature_idx]

print(f"\nMost important feature: {most_important_feature}")


# %%
# **XGB Feature Importances**

#The feature importance scores shown by XGBoost indicate the  
#contribution of each feature to the model's predictions of loan status.
# Get the feature importances from the trained model
feature_importances = xgb_clf.feature_importances_

# Create a dictionary to store the feature names and their importances
feature_importance_dict = dict(zip(feature_names, feature_importances))

# Sort the dictionary by feature importance in descending order
sorted_feature_importance_dict = dict(sorted(feature_importance_dict.items(), key=lambda x: x[1], reverse=True))

# Print the feature importances
print("Feature Importances:")
for feature, importance in sorted_feature_importance_dict.items():
    print(f"{feature}: {importance:.4f}")

# Plot the feature importances
plt.figure(figsize=(10, 6))
plt.bar(range(len(sorted_feature_importance_dict)), list(sorted_feature_importance_dict.values()), align='center')
plt.xticks(range(len(sorted_feature_importance_dict)), list(sorted_feature_importance_dict.keys()), rotation=90)
plt.xlabel('Features')
plt.ylabel('Importance')
plt.title('Feature Importances')
plt.show()

# %%
# **Creating a Shap Plot**

features = [
    'remainder__term', 'encoder__initial_list_status_f', 'remainder__tot_cur_bal', 'remainder__last_week_pay',
    'remainder__int_rate', 'encoder__verification_status_Source Verified', 'encoder__verification_status_Not Verified',
    'encoder__purpose_small_business', 'remainder__dti', 'encoder__verification_status_Verified',
    'remainder__inq_last_6mths', 'encoder__purpose_medical', 'remainder__revol_util', 'encoder__purpose_car',
    'encoder__purpose_debt_consolidation', 'remainder__loan_amnt', 'remainder__annual_inc', 'remainder__total_rec_int',
    'remainder__emp_length', 'encoder__purpose_credit_card', 'remainder__revol_bal', 'encoder__home_ownership_RENT',
    'remainder__open_acc', 'encoder__purpose_major_purchase', 'encoder__purpose_other', 'encoder__purpose_wedding',
    'encoder__home_ownership_MORTGAGE', 'encoder__purpose_moving', 'encoder__home_ownership_OWN', 'encoder__purpose_house',
    'encoder__purpose_home_improvement', 'encoder__purpose_vacation', 'encoder__home_ownership_ANY',
    'encoder__home_ownership_NONE', 'encoder__home_ownership_OTHER', 'encoder__purpose_educational',
    'encoder__purpose_renewable_energy', 'encoder__initial_list_status_w'
]

# Assuming you have trained your XGBoost classifier (xgb_clf) and have the training data (X_train)
explainer = shap.TreeExplainer(xgb_clf)
shap_values = explainer.shap_values(X_train[features])

shap.summary_plot(shap_values, X_train[features], feature_names=features)
# %%
# **Saving our best model, Encoding, and our Scaling**
# Save the trained model
with open('xgb_clf.pkl', 'wb') as file:
    pickle.dump(xgb_clf, file)

# Save the preprocessor object
with open('preprocessor.pkl', 'wb') as file:
    pickle.dump(preprocessor, file)


# Save the scaler object
with open('scaler.pkl', 'wb') as file:
    pickle.dump(scaler, file)
