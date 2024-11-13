import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.model_selection import train_test_split
import os

class DataPreprocessor:
    def __init__(self, file_path):
        self.file_path = file_path
    
    def data_preprocessing(self):
        # Load the dataset
        df = pd.read_csv(self.file_path)
        
        # Drop the 'Loan_ID' column
        df.drop('Loan_ID', axis=1, inplace=True)
        
        # Separate categorical and numerical columns
        categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
        numerical_cols = df.select_dtypes(include=['number']).columns.tolist()
        
        # Columns for one-hot encoding
        one_hot_cols = ['Dependents', 'Property_Area']
        label_cols = ['Gender', 'Married', 'Education', 'Self_Employed']
        target_col = 'Loan_Status'
        
        # Handle numerical and one-hot encoded columns
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', Pipeline([
                    ('imputer', SimpleImputer(strategy='median')),
                    ('scaler', StandardScaler())
                ]), numerical_cols),
                
                ('one_hot', Pipeline([
                    ('imputer', SimpleImputer(strategy='most_frequent')),
                    ('encoder', OneHotEncoder(handle_unknown='ignore'))
                ]), one_hot_cols)
            ],
            remainder='drop'  # Drop columns that are not specified
        )

        # Apply transformations on pipeline features
        transformed_data = preprocessor.fit_transform(df)
        
        # Get OHE feature names after transformation
        one_hot_feature_names = list(preprocessor.named_transformers_['one_hot'].named_steps['encoder'].get_feature_names_out(one_hot_cols))
        feature_names = numerical_cols + one_hot_feature_names
        
        # Create a DataFrame with the transformed data
        transformed_data_df = pd.DataFrame(transformed_data, columns=feature_names)
        
        # Label encode the categorical columns outside of the pipeline
        label_encoder = []
        for col in label_cols:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
            label_encoder.append(le)
        
        # Encode the target variable 'Loan_Status' separately
        target_encoder = LabelEncoder()
        df[target_col] = target_encoder.fit_transform(df[target_col])
        
        # Combine the processed numerical, one-hot encoded, and label-encoded columns
        final_data = pd.concat([transformed_data_df, df[label_cols], df[target_col]], axis=1)
        
        # Save the final DataFrame to a CSV file
        output_path = r'C:\ML Training Code\Loan Predictor\Code\Datasets\processed_data.csv'
        final_data.to_csv(output_path, index=False)
        
        # Return the preprocessor and label encoders
        return preprocessor, label_encoder, target_encoder
    

    def data_split(self):
        df = pd.read_csv('C:/ML Training Code/Loan Predictor/Code/Datasets/processed_data.csv')
        X = df.drop(columns=['Loan_Status'])
        y = df['Loan_Status']
        return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y) # Stratify will help in class imbalance

# Usage example:
# Initialize the LoanDataPreprocessor with the path to the dataset
#loan_preprocessor = DataPreprocessor('C:/ML Training Code/Loan Predictor/Code/Datasets/loan_data 2.csv')

# Preprocess the data and get the preprocessor and label encoders
#preprocessor, le = loan_preprocessor.data_preprocessing()

# Output a message indicating where the processed data is saved
#print(preprocessor)
#print(le)

#print(DataPreprocessor.data_split())
