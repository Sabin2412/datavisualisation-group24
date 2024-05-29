import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Load your data
df = pd.read_csv('/Users/sabinghimire/data visualisation/clean csv/cleaned_retractions35215.csv')

# Initialize label encoder
le = LabelEncoder()

# List of columns to encode
categorical_cols = ['Title', 'Subject', 'Institution', 'Journal', 'Publisher', 'Country', 'Author', 'ArticleType', 'RetractionNature', 'Reason', 'Paywalled']

# Apply label encoder to each categorical column
for col in categorical_cols:
    df[col] = le.fit_transform(df[col])

# Verify changes
print(df.head())


