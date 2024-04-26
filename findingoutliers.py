import pandas as pd

# Load the data
data = pd.read_csv('/Users/sabinghimire/data visualisation/cleaned_retractions35215.csv')

# Select numerical columns
numerical_cols = ['Record ID', 'RetractionDate', 'RetractionPubMedID', 'OriginalPaperDate', 'OriginalPaperPubMedID', 'CitationCount']

# Calculate Z-score for each numerical column
z_scores = pd.DataFrame()
for col in numerical_cols:
    z_scores[col] = (data[col] - data[col].mean()) / data[col].std()

# Threshold for considering outliers
threshold = 3

# Identify outliers using Z-score
outliers = (z_scores > threshold) | (z_scores < -threshold)

# Print columns with outliers
print("Columns with outliers:")
for col in numerical_cols:
    if outliers[col].any():
        print(col)
