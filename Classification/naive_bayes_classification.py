import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix

dataset = pd.read_csv('Datasets/Social_Network_Ads.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Scaling the features

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

print(X_test)

# Train and test the model

model = GaussianNB()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

y_pred = model.predict(X_test)

print(model.predict(sc.transform([[30, 87000]])))

y_pred = y_pred.reshape(len(y_pred),1)
print(y_pred)

y_test= y_test.reshape(len(y_test),1)

print(np.concatenate((y_pred, y_test),1))

print(confusion_matrix(y_test, y_pred))
print(accuracy_score(y_test, y_pred))

