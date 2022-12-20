#!/usr/bin/env python
# coding: utf-8

# # **Cirrhosis Prediction**

# ### **Importing Libraries**

# In[1]:


import pandas as pd
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split


# ### **Importing the Dataset**

# In[2]:


df = pd.read_csv('Dataset/cirrhosis.csv')
df


# ### **Checking Data Types**

# In[3]:


df.dtypes


# ### **Checking NaN in the Dataset**

# In[4]:


df.isnull().sum()


# ### **Correlation**

# In[5]:


plt.figure(figsize = (20, 16))
sns.heatmap(df.corr(), annot = True, cmap = 'BrBG')


# ### **Z-Score**
# 
# To understand how the numerical data deviates from the mean. Z-score is a numerical measurement that describes a value's relationship to the mean of a group of values. If the z-score is 0 - data point's score is identical to the mean score. 
# 
# Given that this data is relatively small, I am trying to keep as many data points. Hence, I will not be removing too many dataset values (unless null for a column wherein I cannot replace it by a rolling mean or another value).

# In[6]:


numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']

num_df = df.select_dtypes(include = numerics)

df_zscore = (num_df - num_df.mean())/num_df.std()

df_zscore


# ### **Data Preprocessing**

# #### Getting Rid of Columns Not Needed

# In[7]:


df = df.drop('ID', axis = 1)
df


# #### Getting Rid of Rows with NaN in Column Drug
# 
# I will be removing the Null Drug data points as I cannot predict the whether D-penicillamine or placebo was admistered to the patient.

# In[8]:


df = df.dropna(subset = 'Drug')
df


# ### **Adding a Rolling Average to Fill in the NaN**

# In[9]:


from sklearn.impute import SimpleImputer

filled_nan = SimpleImputer(missing_values = np.nan, strategy = 'mean')
filled_nan.fit(df.iloc[:, 10:-1])

df.iloc[:, 10:-1] = filled_nan.transform(df.iloc[:, 10:-1])

df


# In[10]:


df.isnull().sum()


# ### **Determining the Age from the N_Days Column**

# In[11]:


df['Age'] = (df['Age'] / 365).round()
df


# ### **One Hot Encoding Categorical Columns**

# In[12]:


# Categorical data columns
categorical_cols = ['Drug', 'Sex', 'Ascites', 'Hepatomegaly', 'Spiders', 'Edema'] 

df_ohe = pd.get_dummies(df, columns = categorical_cols)

df_ohe


# In[13]:


df_ohe.columns


# ### **Encoding the Status (Dependent) into A Column**

# In[14]:


df_ohe['Status'].unique()


# In[15]:


encoded_status = []

for i in range(0, len(df_ohe['Status'])):
    if df_ohe['Status'].iloc[i] == 'D':
        encoded_status.append(0)
    elif df_ohe['Status'].iloc[i] == 'C':
        encoded_status.append(1)
    else:
        encoded_status.append(2)

df_ohe = df_ohe.drop(['Status'], axis = 1)
df_ohe['Status'] = encoded_status

df_ohe


# ### **Splitting the Dataset into Independent and Dependent Variables**

# In[16]:


X = df_ohe.drop(['Status'], axis = 1)
y = df_ohe['Status']


# ### **Split the Dataset into Train and Test Datasets**

# In[17]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 150)


# ### **View Training Datasets**

# In[18]:


X_train


# In[19]:


y_train


# ### **View the Testing Datasets**

# In[20]:


X_test


# In[21]:


y_test


# ### **XGBoost**

# In[22]:


from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold

'''
KFold is a cross-validator that divides the dataset into k folds.
Stratified is to ensure that each fold of dataset has the same proportion of observations with a given label.
'''

from sklearn.model_selection import cross_val_score

xgbc = XGBClassifier()

xgbc.fit(X_train.copy(), y_train.copy())
skfold = StratifiedKFold(n_splits = 8)
xgbc_scores = cross_val_score(estimator = xgbc, X = X_train.copy(), y = y_train.copy(), cv = skfold)
xgbc_scores


# #### Predict using XGBoost

# In[23]:


y_pred_xg = xgbc.predict(X_test.copy())
y_pred_xg


# #### XGBoost Confusion Matrix

# In[24]:


from sklearn.metrics import confusion_matrix
import seaborn as sns

cm_xg = confusion_matrix(y_test, y_pred_xg) 
sns.heatmap(cm_xg, annot = True)


# #### XGBoost Distribution Plot

# In[25]:


sns.distplot(y_test - y_pred_xg)


# ### **Random Forest Classifier**

# In[26]:


from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(n_estimators = 100, criterion = "entropy")

rf.fit(X_train.copy(), y_train.copy())
skfold = StratifiedKFold(n_splits = 8)
rf_scores = cross_val_score(estimator = rf, X = X_train.copy(), y = y_train.copy(), cv = skfold)
rf_scores


# #### Predict Using Random Forest Classification

# In[27]:


y_pred_rf = rf.predict(X_test.copy())
y_pred_rf


# #### Random Forest Classification Confusion Matrix

# In[28]:


from sklearn.metrics import confusion_matrix
import seaborn as sns

cm_rf = confusion_matrix(y_test, y_pred_rf) 
sns.heatmap(cm_rf, annot = True)


# #### Random Forest Classifier Distribution Plot

# In[29]:


sns.distplot(y_test - y_pred_rf)


# ### **Comparison: XGBoost VS Random Forest Classifier**

# In[30]:


from sklearn.metrics import mean_absolute_error

print(f"XGBoost Mean Score : {mean_absolute_error(y_test, y_pred_xg)}")
print(f"XGBoost Accuracy : {xgbc_scores.mean() * 100} %\n")

print(f"Random Forest Mean Score : {mean_absolute_error(y_test, y_pred_rf)}")
print(f"Random Forest Accuracy : {rf_scores.mean() * 100} %\n")


# In[31]:


from sklearn.metrics import classification_report

print('\nXGBoost Classification Report: \n', classification_report(y_test, y_pred_xg))
print('\nRandom Forest Classification Report: \n', classification_report(y_test, y_pred_rf))


# In[ ]:




