#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


import os
print(os.listdir("."))


# In[3]:


#Read CSV file
data = pd.read_csv("data.csv")


# In[4]:


#Understand the data
data.head()


# In[5]:


data.columns


# In[6]:


#Remove unecessary data
data = data.drop(['Unnamed: 32', 'id'], axis = 1)

#Create my label vector
y = data.diagnosis

#Create my feature matrix
data = data.drop('diagnosis', axis = 1)
x = data
x.head()


# In[7]:


#Get to know the data
histogram = sns.countplot(y)
B, M = y.value_counts()
print('Number of Benign: ', B)
print('Number of Malignant: ', M)


# In[8]:


x.describe()


# In[9]:


data_dia = y
data = x
data_n_2 = (data - data.mean()) / (data.std())              # standardization


# In[10]:


#Draw violin plot
def violinplot(start, end):
    data = pd.concat([y,data_n_2.iloc[:,start:end]],axis=1)
    data = pd.melt(data,id_vars="diagnosis", var_name="features", value_name='value')
    plt.figure(figsize=(10,10))
    sns.violinplot(x="features", y="value", hue="diagnosis", data=data,split=True, inner="quart")
    plt.xticks(rotation=90)


# In[11]:


violinplot(0,10)
violinplot(10,20)
violinplot(20,31)


# In[12]:


#Draw boxplot
def boxplot(start, end):
    data = pd.concat([y,data_n_2.iloc[:,start:end]],axis=1)
    data = pd.melt(data,id_vars="diagnosis", var_name="features", value_name='value')
    plt.figure(figsize=(8,8))
    sns.boxplot(x="features", y="value", hue="diagnosis", data=data)
    plt.xticks(rotation=90)


# In[13]:


boxplot(0,10) #mean values
boxplot(10,20) #se
boxplot(20,31) #worst


# In[14]:


#Draw joint plot
sns.jointplot(x.loc[:,'concavity_worst'], x.loc[:,'concave points_worst'], kind="regg", color="#ce1414")


# In[15]:


sns.set(style="whitegrid", palette="muted")
data_dia = y
data = x
data_n_2 = (data - data.mean()) / (data.std())  #Standardize


# In[16]:


#Draw swarm Plot
def swarmPlot(start, end):
    data = pd.concat([y,data_n_2.iloc[:,start:end]],axis=1)
    data = pd.melt(data,id_vars="diagnosis",
                    var_name="features",
                    value_name='value')
    plt.figure(figsize=(10,10))
    sns.swarmplot(x="features", y="value", hue="diagnosis", data=data)
    plt.xticks(rotation=90)


# In[17]:


swarmPlot(0,10)
swarmPlot(10,20)
swarmPlot(20,31)


# In[18]:


X = (x.values)

X_average_row = np.average(X, axis=0)
X_bar = X - X_average_row
sigma = X_bar.T.dot(X_bar)

eigenvalues, _ = np.linalg.eig(sigma)

plt.plot(np.arange(len(eigenvalues)), eigenvalues)
plt.show()


# In[19]:


plt.plot(np.arange(7), eigenvalues[:7])
axes = plt.gca()
axes.set_xlim([0, 7])
axes.set_ylim([0,10e5])
plt.show()


# In[20]:


#Drop "bad features" and test others using correlation heat map
possible_features = x.drop(['texture_mean', 'smoothness_mean', 'symmetry_mean', 'fractal_dimension_mean', 'texture_se', 'smoothness_se', 'symmetry_se', 'fractal_dimension_se', 'texture_worst', 'smoothness_worst', 'symmetry_worst', 'fractal_dimension_worst'], axis = 1)
possible_features_1 = possible_features.iloc[:,:6]
possible_features_2 = possible_features.iloc[:,6:12]
possible_features_3 = possible_features.iloc[:,12:]

print(possible_features_1.columns)
print(possible_features_2.columns)
print(possible_features_3.columns)


# In[21]:


f,ax = plt.subplots(figsize=(8, 8))
sns.heatmap(possible_features_1.corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax)

f,ax = plt.subplots(figsize=(8, 8))
sns.heatmap(possible_features_2.corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax)

f,ax = plt.subplots(figsize=(8, 8))
sns.heatmap(possible_features_3.corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax)


# In[22]:


drop_list_RandF = ['area_mean', 'radius_mean', 'compactness_mean', 'area_se', 'radius_se', 'compactness_se', 'area_worst', 'radius_worst', 'compactness_worst' ]
x_1 = x.drop(drop_list_RandF, axis = 1)
x_1.head()


# In[23]:


#Heat map to check correlation values
f,ax = plt.subplots(figsize=(10, 10))
sns.heatmap(x_1.corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax)


# In[31]:


#Random Forest model
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import f1_score,confusion_matrix
from sklearn.metrics import accuracy_score

# Split 70% train 30% test
x_train, x_test, y_train, y_test = train_test_split(x_1, y, test_size=0.3) #random_state=20

# Create the parameter grid based on the results of random search 
param_grid = {
    'bootstrap': [True],
    'max_depth': [5, 8, 12],
    'max_features': [1, 2, "auto"],
    'min_samples_leaf': [1, 2, 3],
    'min_samples_split': [2, 3],
    'n_estimators': [8, 10, 12, 14]
}

#Create the model to tune
clf_rf = RandomForestClassifier()
clf_gridsearch = GridSearchCV(estimator = clf_rf, param_grid = param_grid, cv = 3, verbose=1, n_jobs=2)

#Fit the model
clr_gridsearch = clf_gridsearch.fit(x_train,y_train)

#Predict
ac_t = accuracy_score(y_train,clr_gridsearch.predict(x_train))
print('Training Accuracy is: ',ac_t)
ac = accuracy_score(y_test,clr_gridsearch.predict(x_test))
print('Test Accuracy is: ',ac)
cm = confusion_matrix(y_test,clr_gridsearch.predict(x_test))
sns.heatmap(cm,annot=True,fmt="d")


# In[ ]:




