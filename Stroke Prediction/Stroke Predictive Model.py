#!/usr/bin/env python
# coding: utf-8

# # **Stroke Prediction**

# ### **Installing Libraries**

# In[96]:


import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns


# ### **Importing the Dataset**

# In[97]:


df = pd.read_csv('Dataset/stroke-dataset.csv')
df


# ### **Data Correlation**

# In[126]:


df.corr()


# In[127]:


sns.heatmap(df.corr(), annot=True, cmap='twilight_shifted')


# ### **Data Preprocessing**

# #### Determining Empty Values

# In[98]:


df.isnull().sum()


# #### Replacing Empty Values with Rolling Average - BMI

# In[99]:


# Finding the rolling average of the BMI
df['Rolling Avg - BMI'] = df['bmi'].rolling(window = 5, min_periods = 1).mean()

# Filling the rows with empty BMI values with the rolling average
df['bmi'] = df['bmi'].fillna(df['Rolling Avg - BMI'])

# Droping the Rolling Average column made - Purpose has been served
df.drop(['Rolling Avg - BMI'], axis = 1, inplace = True)

# Checking whether an NaN exist within the DataFrame
df.isnull().sum()


# ### **Handling Outliers**

# #### Visualize Analysis

# In[100]:


# Average Glucose Level
sns.boxplot(x = df['avg_glucose_level'])
plt.show()


# In[101]:


# BMI
sns.boxplot(x = df['bmi'])
plt.show()


# #### Numerical Analysis

# In[102]:


df[['avg_glucose_level', 'bmi']].describe()


# #### Calculating the Outliers

# In[103]:


Q1_glucose = df['avg_glucose_level'].quantile(0.25)
Q3_glucose = df['avg_glucose_level'].quantile(0.75)

Q1_bmi = df['bmi'].quantile(0.25)
Q3_bmi = df['bmi'].quantile(0.75)

IQR_glucose = Q3_glucose - Q1_glucose
IQR_bmi = Q3_bmi - Q1_bmi


# In[104]:


lower_lim_glucose = Q1_glucose - (1.5 * IQR_glucose)
upper_lim_glucose = Q1_glucose + (1.5 * IQR_glucose)

lower_lim_bmi = Q1_bmi - (1.5 * IQR_bmi)
upper_lim_bmi = Q1_bmi + (1.5 * IQR_bmi)


# #### Dropping the Rows with Outliers

# In[105]:


outlier_glucose_low = (df['avg_glucose_level'] < lower_lim_glucose)
outlier_glucose_high = (df['avg_glucose_level'] > upper_lim_glucose)

outlier_bmi_low = (df['bmi'] < lower_lim_bmi)
outlier_bmi_high = (df['bmi'] > upper_lim_bmi)


# In[106]:


df = df[~(outlier_glucose_low | outlier_glucose_high)]
df = df[~(outlier_bmi_low | outlier_bmi_high)]

df


# #### Visual Analysis After Dropping the Outliers

# In[107]:


# Average Glucose Level
sns.boxplot(x = df['avg_glucose_level'])
plt.show()


# In[108]:


# BMI
sns.boxplot(x = df['bmi'])
plt.show()


# ### **One Hot Encoding the Columns with Categorical Data**

# #### Gender

# In[109]:


ohe_gender = pd.get_dummies(df.gender, prefix='gender')
ohe_gender


# #### Marriage

# In[110]:


ohe_ever_married = pd.get_dummies(df.ever_married, prefix='ever_married')
ohe_ever_married


# #### Work Type

# In[111]:


ohe_work_type = pd.get_dummies(df.work_type, prefix='work_type')
ohe_work_type


# #### Residence

# In[112]:


ohe_residence = pd.get_dummies(df.Residence_type, prefix='Residence_type')
ohe_residence


# #### Smoking

# In[113]:


ohe_smoking = pd.get_dummies(df.smoking_status, prefix='smoking_status')
ohe_smoking


# ### **Split Dataset into Independent & Dependent Variables**

# In[114]:


frames = [df['id'], ohe_gender, df['age'], df['hypertension'], df['heart_disease'], ohe_ever_married, ohe_work_type, ohe_residence, df['avg_glucose_level'], df['bmi'], ohe_smoking]

X = pd.concat(frames, axis = 1, join = 'inner')
y = df['stroke']


# ### **Splitting X and y into Train and Test Datasets**

# In[115]:


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 100)


# ### **Checking the Training Datasets**

# In[116]:


X_train


# In[117]:


y_train


# ### **Checking the Testing Datasets**

# In[118]:


X_test


# In[119]:


y_test


# ### **Making the SVM Classification Model**

# In[120]:


from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

clf = make_pipeline(StandardScaler(), SVC(gamma = 'auto', kernel = 'poly'))
clf.fit(X, y)


# ### **Predicting Using the Model Created**

# In[121]:


y_pred = clf.predict(X_test)
y_pred


# ### **Comparing Results of Y Test and Y Predicted**

# In[122]:


y_grid = (np.column_stack([y_test, y_pred]))
y_grid


# ### **Accuracy Score**

# In[123]:


from sklearn.metrics import accuracy_score
accuracy_score(y_test, y_pred)


# ### **Confusion Matrix**

# In[124]:


from sklearn import metrics

cm = metrics.confusion_matrix(y_test, y_pred)

df_cm = pd.DataFrame(cm)

plt.figure(figsize = (10,8))
sns.heatmap(df_cm, annot=True ,fmt='g')


# ### **Plot the Difference in Results**

# In[125]:


import plotly.graph_objects as go

X_test_len = []
for i in range(0, len(X_test)):
    X_test_len.append(i + 1)

fig = go.Figure()

fig.add_trace(
    go.Scatter(
        x = X_test_len,
        y = y_test,
        mode = 'markers+lines',
        name = 'Test Dataset',
    ))

fig.add_trace(
    go.Scatter(
        x = X_test_len,
        y = y_pred,
        mode = 'markers+lines',
        name = 'Predicted Dataset'
    ))

fig.update_layout(
    title = "Comparing Y Test Versus Y Predicted",
    xaxis_title = "Patient",
    yaxis_title = "Stroke Identification",
    legend_title = "Output Datasets",
    font=dict(
        family = "Times New Roman, monospace",
        size = 15,
        color = "RebeccaPurple"
    )
)

fig.show()

