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




