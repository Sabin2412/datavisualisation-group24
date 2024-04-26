import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
data = pd.read_csv('/Users/neha/data visualisation/cleaned_retractions35215.csv')

# Ensure 'RetractionDate' is a datetime type
data['RetractionDate'] = pd.to_datetime(data['RetractionDate'])

# Extract the year from 'RetractionDate' for annual trend analysis
data['Year'] = data['RetractionDate'].dt.year

# Group by both 'Country' and 'Journal'
country_journal_group = data.groupby(['Country', 'Journal']).size().unstack(fill_value=0)

# Filter for visualization clarity
min_retractions = 5
country_journal_group = country_journal_group.loc[:, (country_journal_group.sum() > min_retractions)]
country_journal_group = country_journal_group.loc[(country_journal_group.sum(axis=1) > min_retractions), :]

# Use a heatmap to visualize the country and journal distribution
plt.figure(figsize=(12, 10))
heatmap = sns.heatmap(country_journal_group, cmap='viridis', linewidths=.5, annot=True, fmt="d")
plt.title('Retractions per Country and Journal')
plt.xlabel('Journal')
plt.ylabel('Country')
plt.xticks(rotation=45)

# Save the heatmap to a file
plt.savefig('/Users/neha/data visualisation/clean csv', format='png', dpi=300)  # Adjust path as needed
plt.close()  # Close the plot to free up memory

print("Heatmap saved successfully!")
