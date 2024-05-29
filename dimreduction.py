import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import Pipeline
from scipy.sparse import hstack

# Load the dataset
df = pd.read_csv('reduced.csv')

# Check for missing values and drop them if any
df.dropna(inplace=True)

# Identify numerical and categorical columns
numerical_columns = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
categorical_columns = df.select_dtypes(include=['object']).columns.tolist()

# Define the preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_columns),
        ('cat', OneHotEncoder(sparse_output=True), categorical_columns)  # Keep sparse to manage memory
    ]
)

# Apply the preprocessing pipeline
X_preprocessed = preprocessor.fit_transform(df)

# Initialize TruncatedSVD
n_components = 3  # Adjust this number based on your needs
svd = TruncatedSVD(n_components=n_components)

# Fit and transform the data
reduced_data = svd.fit_transform(X_preprocessed)

# Create a DataFrame with the reduced data and descriptive column names
reduced_df = pd.DataFrame(reduced_data, columns=[f'SVD_Component_{i+1}' for i in range(n_components)])

# Optionally, keep some of the original columns in the final DataFrame
# For example, here we keep the first few original columns:
original_columns_to_keep = numerical_columns[:min(5, len(numerical_columns))]  # Adjust as needed
final_df = pd.concat([df[original_columns_to_keep].reset_index(drop=True), reduced_df], axis=1)

# Save the reduced data to a new CSV file
final_df.to_csv('reduced_data_with_original.csv', index=False)

print("Dimensionality reduction complete. The reduced dataset with original columns is saved as 'reduced_data_with_original.csv'.")
