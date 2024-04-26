import pandas as pd
import matplotlib.pyplot as plt

# Load your dataset
data = pd.read_csv('/Users/sabinghimire/data visualisation/clean csv/cleaned_retractions35215.csv')

# Replace missing values in 'Paywalled' column with a placeholder value, e.g., 'Unknown'
data['Paywalled'].fillna('Unknown', inplace=True)

# Count the occurrences of paywalled and non-paywalled articles
paywalled_counts = data['Paywalled'].value_counts()

# Plot a pie chart
plt.figure(figsize=(8, 8))
paywalled_counts.plot(kind='pie', autopct='%1.1f%%', startangle=140)
plt.title('Paywalled vs Non-paywalled Articles')
plt.ylabel('')  # No need to display the ylabel
plt.legend(loc='best')
plt.show()
