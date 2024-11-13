from data_processing import DataPreprocessor
from model_training import ModelTrainer
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

class ModelEvaluator():
    def __init__(self):
        pass

    def evaluate_models(self, trained_ann, trained_stack, X_train, X_test, y_train, y_test):
        # Evaluate the ANN model on training data
        y_train_pred_ann = (trained_ann.predict(X_train) > 0.5).astype("int32")
        ann_train_accuracy = accuracy_score(y_train, y_train_pred_ann)
        ann_train_conf_matrix = confusion_matrix(y_train, y_train_pred_ann)
        ann_train_false_positives = ann_train_conf_matrix[0][1]
        ann_train_precision = precision_score(y_train, y_train_pred_ann)
        ann_train_recall = recall_score(y_train, y_train_pred_ann)
        ann_train_f1 = f1_score(y_train, y_train_pred_ann)

        # Evaluate the ANN model on testing data
        y_test_pred_ann = (trained_ann.predict(X_test) > 0.5).astype("int32")
        ann_test_accuracy = accuracy_score(y_test, y_test_pred_ann)
        ann_test_conf_matrix = confusion_matrix(y_test, y_test_pred_ann)
        ann_test_false_positives = ann_test_conf_matrix[0][1]
        ann_test_precision = precision_score(y_test, y_test_pred_ann)
        ann_test_recall = recall_score(y_test, y_test_pred_ann)
        ann_test_f1 = f1_score(y_test, y_test_pred_ann)

        # Evaluate the stacking classifier on training data
        y_train_pred_stacking = trained_stack.predict(X_train)
        stacking_train_accuracy = accuracy_score(y_train, y_train_pred_stacking)
        stacking_train_conf_matrix = confusion_matrix(y_train, y_train_pred_stacking)
        stacking_train_false_positives = stacking_train_conf_matrix[0][1]
        stacking_train_precision = precision_score(y_train, y_train_pred_stacking)
        stacking_train_recall = recall_score(y_train, y_train_pred_stacking)
        stacking_train_f1 = f1_score(y_train, y_train_pred_stacking)

        # Evaluate the stacking classifier on testing data
        y_test_pred_stacking = trained_stack.predict(X_test)
        stacking_test_accuracy = accuracy_score(y_test, y_test_pred_stacking)
        stacking_test_conf_matrix = confusion_matrix(y_test, y_test_pred_stacking)
        stacking_test_false_positives = stacking_test_conf_matrix[0][1]
        stacking_test_precision = precision_score(y_test, y_test_pred_stacking)
        stacking_test_recall = recall_score(y_test, y_test_pred_stacking)
        stacking_test_f1 = f1_score(y_test, y_test_pred_stacking)

        # Prepare evaluation results
        evaluation_results = (

            "ANN Model Training Evaluation:\n"
            f"Accuracy: {ann_train_accuracy:.4f}\n"
            f"False Positives: {ann_train_false_positives}\n"
            f"Precision: {ann_train_precision:.4f}\n"
            f"Recall: {ann_train_recall:.4f}\n"
            f"F1 Score: {ann_train_f1:.4f}\n\n"

            "ANN Model Testing Evaluation:\n"
            f"Accuracy: {ann_test_accuracy:.4f}\n"
            f"False Positives: {ann_test_false_positives}\n"
            f"Precision: {ann_test_precision:.4f}\n"
            f"Recall: {ann_test_recall:.4f}\n"
            f"F1 Score: {ann_test_f1:.4f}\n\n"


            "Stacking Model Training Evaluation:\n"
            f"Accuracy: {stacking_train_accuracy:.4f}\n"
            f"False Positives: {stacking_train_false_positives}\n"
            f"Precision: {stacking_train_precision:.4f}\n"
            f"Recall: {stacking_train_recall:.4f}\n"
            f"F1 Score: {stacking_train_f1:.4f}\n\n"

            "Stacking Model Testing Evaluation:\n"
            f"Accuracy: {stacking_test_accuracy:.4f}\n"
            f"False Positives: {stacking_test_false_positives}\n"
            f"Precision: {stacking_test_precision:.4f}\n"
            f"Recall: {stacking_test_recall:.4f}\n"
            f"F1 Score: {stacking_test_f1:.4f}\n"

        )

        # Save evaluation metrics to a text file
        with open("Model Evaluation Final.txt", "w") as f:
            f.write(evaluation_results)

        # Decide the best model based on fewer false positives on the test set
        if stacking_test_false_positives < ann_test_false_positives:
            best_model = trained_stack
            print("Stacking model is selected as the best model.")
        elif stacking_test_false_positives > ann_test_false_positives:
            best_model = trained_ann
            print("ANN model is selected as the best model.")
        else:
            # If both have the same number of false positives, choose the one with greater accuracy
            if stacking_test_accuracy > ann_test_accuracy:
                best_model = trained_stack
                print("Both models have the same false positives. Stacking model is selected due to higher accuracy.")
            else:
                best_model = trained_ann
                print("Both models have the same false positives. ANN model is selected due to higher accuracy.")

        return best_model

# obj1 = DataPreprocessor('C:/ML Training Code/Loan Predictor/Code/Datasets/loan_data 2.csv')
# X_train, X_test, y_train, y_test = obj1.data_split()

# obj2 = ModelTrainer()
# trained_ann, trained_stack = obj2.train_models(X_train, y_train)

# obj3 = ModelEvaluator()
# obj3.evaluate_models(trained_ann, trained_stack, X_train, X_test, y_train, y_test)

