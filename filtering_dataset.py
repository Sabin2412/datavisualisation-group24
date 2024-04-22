import pandas as pd

# Loading  the  given csv file 
data = pd.read_csv('retractions35215.csv')  

# Converting  date columns to datetime format, handling inconsistent formats
data['RetractionDate'] = pd.to_datetime(data['RetractionDate'], errors='coerce')
data['OriginalPaperDate'] = pd.to_datetime(data['OriginalPaperDate'], errors='coerce')

# Handling missing values by assigning the results back to the DataFrame
data.dropna(subset=['RetractionDOI', 'Institution'], inplace=True)
data['URLS'] = data['URLS'].fillna('Not Available')
data['Notes'] = data['Notes'].fillna('Not Available')
data['RetractionPubMedID'] = data['RetractionPubMedID'].fillna(0)
data['OriginalPaperPubMedID'] = data['OriginalPaperPubMedID'].fillna(0)
most_common_paywalled = data['Paywalled'].mode()[0]
data['Paywalled'] = data['Paywalled'].fillna(most_common_paywalled)

# Saveing  the cleaned data to a new CSV file in the same directory
data.to_csv('cleaned_retractions35215.csv', index=False)

print("Data cleaned and saved successfully!")
