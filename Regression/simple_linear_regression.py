import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

matplotlib.use('Agg')  # Use a non-interactive backend

# Importing the dataset
dataset = pd.read_csv('Datasets\Salary_Data.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 32)

# Training the model and prediction  
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Output visualization 

# Training viz : 

plt.scatter(X_train, y_train , color = 'red')
plt.plot(X_train, model.predict(X_train), color = 'blue')
plt.title('Training results')
plt.xlabel('YOE')
plt.ylabel('Salary')
plt.show()

