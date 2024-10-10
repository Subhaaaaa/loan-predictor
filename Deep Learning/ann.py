import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, confusion_matrix



# Data preprocessing 

dataset = pd.read_csv('Datasets/Churn_modelling.csv')
X = dataset.iloc[:, 3:-1].values
y = dataset.iloc[:, -1].values
print(X)


# Encoding 
## Binary variables are label encoded
encoder = LabelEncoder()
y = encoder.fit_transform(y)
X[:, 2] = encoder.fit_transform(X[:, 2])
print(X[:, 2], y)

## One hot encoding for geography column
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [1])], remainder='passthrough')
X = np.array(ct.fit_transform(X))
print(type(X))

#Splitting into train test 

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Feature scaling is really important in deeplearning and is a must step
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)

# Creating an ANN
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Dense(units=6, activation = 'relu')) # First hidden layer
model.add(tf.keras.layers.Dense(units=6, activation = 'relu')) # Second hidden layer
model.add(tf.keras.layers.Dense(units=1, activation = 'relu')) # Output layer

# Compiling the ANN
model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Train the ANN 
model.fit(X_train, y_train, batch_size = 32, epochs = 50)
print(model)


y_pred = model.predict(X_test)
y_pred = y_pred > 0.5

y_pred = y_pred.reshape(len(y_pred),1)
print(y_pred)

y_test= y_test.reshape(len(y_test),1)

print(np.concatenate((y_pred, y_test),1))

print(confusion_matrix(y_test, y_pred))
print(accuracy_score(y_test, y_pred))



