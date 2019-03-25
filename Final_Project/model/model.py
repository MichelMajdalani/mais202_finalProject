#!/usr/bin/env python
# coding: utf-8

# In[1]:

# Import bread and butter packages
import numpy as np
import pandas as pd
from joblib import dump, load

# Random Forest model
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score

# In[3]:

# Read CSV file
data = pd.read_csv("data.csv")

# In[6]:

# Remove unecessary data
data = data.drop(['Unnamed: 32', 'id'], axis=1)

# Create my label vector
y = data.diagnosis

# Create my feature matrix
data = data.drop('diagnosis', axis=1)
x = data


# In[9]:

data_dia = y
data = x
data_n_2 = (data - data.mean()) / (data.std())              # standardization


X = (x.values)

X_average_row = np.average(X, axis=0)
X_bar = X - X_average_row
sigma = X_bar.T.dot(X_bar)

eigenvalues, _ = np.linalg.eig(sigma)

x_1 = x.loc(axis=1)['area_mean', 'area_se', 'area_worst', 'compactness_mean', 'compactness_se',
                    'compactness_worst', 'texture_mean', 'texture_se', 'texture_worst', 'smoothness_mean', 'symmetry_mean']

# In[31]:

# Split 70% train 30% test
x_train, x_test, y_train, y_test = train_test_split(x_1, y, test_size=0.3)  # random_state=20

# Create the parameter grid based on the results of random search
param_grid = {
    'bootstrap': [True],
    'max_depth': [5, 8, 12],
    'max_features': [1, 2, "auto"],
    'min_samples_leaf': [1, 2, 3],
    'min_samples_split': [2, 3],
    'n_estimators': [8, 10, 12, 14]
}

# Create the model to tune
clf_rf = RandomForestClassifier()
clf_gridsearch = GridSearchCV(estimator=clf_rf, param_grid=param_grid, cv=3, verbose=1, n_jobs=2)

# Fit the model
model = clf_gridsearch.fit(x_train, y_train)

# Save model
dump(model, 'model.joblib')

# Predict
y_pred = model.predict(x_test)
ac_t = accuracy_score(y_train, model.predict(x_train))
ac = accuracy_score(y_test, y_pred)
print(ac)


# In[ ]:
