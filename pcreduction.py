import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np
import joblib

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
        ('cat', OneHotEncoder(sparse_output=False), categorical_columns)  # Use dense output for PCA
    ]
)

# Apply the preprocessing pipeline
X_preprocessed = preprocessor.fit_transform(df)

# Initialize PCA
n_components = 15  # Number of principal components to keep
pca = PCA(n_components=n_components)

# Fit and transform the data
reduced_data = pca.fit_transform(X_preprocessed)

# Create a DataFrame with the reduced data and descriptive column names
reduced_df = pd.DataFrame(reduced_data, columns=[f'PC{i+1}' for i in range(n_components)])

# Retain specific original columns
columns_to_retain = [
    'Record ID', 'Retraction DOI', 'Original Paper DOI', 'Journal',
    'Retraction Date', 'Notes', 'Reason'
]

# Combine retained original columns with PCA components
final_df = pd.concat([df[columns_to_retain].reset_index(drop=True), reduced_df], axis=1)

# Save the reduced data to a new CSV file
final_df.to_csv('reduced_data_with_original_columns.csv', index=False)

print("Dimensionality reduction complete. The reduced dataset with specified original columns is saved as 'reduced_data_with_original_columns.csv'.")

# Save the PCA model
joblib.dump(pca, 'pca_model.pkl')
print("PCA model saved as 'pca_model.pkl'.")

# Explained variance
explained_variance = pca.explained_variance_ratio_
cumulative_explained_variance = np.cumsum(explained_variance)

# Check the explained variance by the first 15 PCs
variance_by_first_15_pcs = cumulative_explained_variance[-1]  # Sum of variance explained by the first 15 PCs
print(f"Explained variance by the first 15 PCs: {variance_by_first_15_pcs * 100:.2f}%")

# Visualize the explained variance
plt.figure(figsize=(10, 5))
plt.bar(range(1, len(explained_variance) + 1), explained_variance, alpha=0.7, align='center', label='Individual explained variance')
plt.step(range(1, len(cumulative_explained_variance) + 1), cumulative_explained_variance, where='mid', label='Cumulative explained variance')
plt.ylabel('Explained variance ratio')
plt.xlabel('Principal components')
plt.legend(loc='best')
plt.title('Explained Variance by Principal Components')
plt.savefig('explained_variance.png')  # Save the plot
plt.show()

# Save the cumulative explained variance plot
plt.figure(figsize=(10, 5))
plt.step(range(1, len(cumulative_explained_variance) + 1), cumulative_explained_variance, where='mid', label='Cumulative explained variance', color='orange')
plt.ylabel('Cumulative explained variance ratio')
plt.xlabel('Principal components')
plt.legend(loc='best')
plt.title('Cumulative Explained Variance by Principal Components')
plt.savefig('cumulative_explained_variance.png')  # Save the plot
plt.show()
