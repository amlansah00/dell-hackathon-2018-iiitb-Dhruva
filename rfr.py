#libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:3].values
y = dataset.iloc[:, 3:5].values

from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators = 10, random_state = 0)
regressor.fit(X, y)

# Predicting a new result
y_pred = regressor.predict(6.5)
