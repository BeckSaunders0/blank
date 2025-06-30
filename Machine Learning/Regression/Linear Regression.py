import numpy as np
from sklearn.datasets import load_diabetes
import pandas as pd
import random

diabetes_data = load_diabetes()
data = pd.DataFrame(data=diabetes_data.data, columns=diabetes_data.feature_names)
num_features = len(data.columns)
n = data.shape[0]
# Index(['age', 'sex', 'bmi', 'bp', 's1', 's2', 's3', 's4', 's5', 's6'], dtype='object')
y = np.array(diabetes_data.target).reshape((442,1))

betas = np.zeros(num_features+1).reshape((11,1)) #Initial values for params
X = np.array(data)

ones = np.ones((n, 1))
X = np.hstack((ones, X))

grad = -2*X.T @ (y - X@betas)

num_epochs = 20
gamma = 0.001

for epoch in range(num_epochs):
    betas = np.array([betas[k] - gamma*grad[k] for k in range(len(grad))])
    grad = -2*X.T @ (y - X@betas)

print(betas)