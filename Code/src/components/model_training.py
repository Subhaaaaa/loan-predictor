from data_processing import DataPreprocessor
from tensorflow import keras
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import StackingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, precision_score, recall_score, f1_score
from tensorflow import keras
from keras import layers

class ModelTrainer():
    def __init__(self):
        pass

    def ann_model(self):
        ann = keras.Sequential()
        ann.add(layers.Dense(64, activation='relu'))
        ann.add(layers.Dense(32, activation='relu'))
        ann.add(layers.Dense(32, activation='relu'))
        ann.add(layers.Dense(1, activation='sigmoid'))  # For binary classification
        ann.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        return ann

    def stacking_classifier(self):
        stack = StackingClassifier(
            estimators=[
                ('log_reg', LogisticRegression()),
                ('naive_bayes', GaussianNB()),
                ('kernel_svm', SVC(kernel='rbf', probability=True)),
                ('random_forest', RandomForestClassifier()),
                ('xgboost', XGBClassifier())
            ],
            final_estimator=SVC(probability=True),
            cv=5
        )
        return stack

    def train_models(self, X_train, y_train):
        ann = ModelTrainer.ann_model(self)
        stack = ModelTrainer.stacking_classifier(self)
        ann.fit(X_train, y_train, epochs=30, verbose=0)
        stack.fit(X_train, y_train)
        return  ann, stack
    
