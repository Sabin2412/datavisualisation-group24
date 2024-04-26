import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load your dataset (already cleaned from outliers)
data = pd.read_csv('/Users/sabinghimire/data visualisation/final_retractions35215.csv')

# Ensure 'RetractionDate' is a datetime format for extracting the year
data['RetractionDate'] = pd.to_datetime(data['RetractionDate'], errors='coerce')

# Create a 'Year' column from 'RetractionDate'
data['Year'] = data['RetractionDate'].dt.year

# Descriptive statistics for 'CitationCount'
mean_value = data['CitationCount'].mean()
median_value = data['CitationCount'].median()
std_dev = data['CitationCount'].std()

print(f"Mean Citation Count: {mean_value}")
print(f"Median Citation Count: {median_value}")
print(f"Standard Deviation of Citation Count: {std_dev}")

# Histogram of citation counts
sns.histplot(data['CitationCount'].dropna())
plt.title('Distribution of Citation Counts')
plt.xlabel('Citation Count')
plt.ylabel('Frequency')
plt.show()

# Box plot for citation counts by year
sns.boxplot(x='Year', y='CitationCount', data=data)
plt.title('Citation Counts by Year')
plt.xlabel('Year')
plt.ylabel('Citation Count')
plt.xticks(rotation=45)  # Rotate x labels for better readability if needed
plt.show()