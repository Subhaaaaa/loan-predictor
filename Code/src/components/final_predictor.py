import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from data_processing import DataPreprocessor
from model_training import ModelTrainer
from model_evaluation import ModelEvaluator

class LoanPredictor():
    
    def __init__(self):
        pass


    def main(self):
        final_predictor, preprocessor, label_encoder, target_encoder = LoanPredictor.initializer() 

        df = pd.read_csv('C:\ML Training Code\Loan Predictor\Code\Datasets\Input_file.csv')
        df = df.drop('Loan_ID', axis=1)
        #print(df.shape)
        numerical_cols = df.select_dtypes(include=['number']).columns.tolist()
        one_hot_cols = ['Dependents', 'Property_Area']
        label_cols = ['Gender', 'Married', 'Education', 'Self_Employed']

        transformed_data = preprocessor.transform(df)

        one_hot_feature_names = list(preprocessor.named_transformers_['one_hot'].named_steps['encoder'].get_feature_names_out(one_hot_cols))
        numerical_cols = [col for col in numerical_cols if col not in one_hot_cols]
        feature_names = numerical_cols + one_hot_feature_names

        transformed_data_df = pd.DataFrame(transformed_data, columns=feature_names)

        le_index = 0
        for col in label_cols:
            df[col] = label_encoder[le_index].transform(df[col])
            le_index += 1
        
        final_data = pd.concat([transformed_data_df, df[label_cols]], axis=1)
        predictions_probabilities = final_predictor.predict(final_data)
        predictions  = (predictions_probabilities > 0.5).astype(int) 

        # Inverse transform the predictions to original labels
        original_predictions = target_encoder.inverse_transform(predictions)

        # Display the output
        df['Predicted_Loan_Status'] = original_predictions
        print("Prediction Results:")
        print(df[['Predicted_Loan_Status']])
        

    @staticmethod
    def initializer():
        data_preprocessor = DataPreprocessor('C:/ML Training Code/Loan Predictor/Code/Datasets/loan_data 2.csv')
        preprocessor, label_encoder, target_encoder = data_preprocessor.data_preprocessing()
        X_train, X_test, y_train, y_test =  data_preprocessor.data_split()

        model_trainer = ModelTrainer()
        trained_ann_classifier, trained_stacked_classifier = model_trainer.train_models(X_train, y_train)

        model_evaluator = ModelEvaluator()
        final_predictor = model_evaluator.evaluate_models(trained_ann_classifier, trained_stacked_classifier, X_train, X_test, y_train, y_test)

        return final_predictor, preprocessor, label_encoder, target_encoder

if __name__ == '__main__':
    lp = LoanPredictor()
    lp.main()
