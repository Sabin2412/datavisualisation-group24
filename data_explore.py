import pandas as pd
import matplotlib.pyplot as plt

# Load the dataset
data = pd.read_csv('/Users/sabinghimire/data visualisation/cleaned_retractions35215.csv')

# Histogram of CitationCount
plt.figure(figsize=(10, 6))
plt.hist(data['CitationCount'], bins=50, color='blue', alpha=0.7)
plt.title('Distribution of Citation Counts')
plt.xlabel('Citation Count')
plt.ylabel('Frequency')
plt.grid(True)
plt.savefig('CitationCount_Histogram.png')  # Saves the plot as a PNG file
plt.close()

# Bar chart of top 10 journals by number of retractions
plt.figure(figsize=(12, 8))
data['Journal'].value_counts().head(10).plot(kind='bar', color='green')
plt.title('Top 10 Journals by Number of Retractions')
plt.xlabel('Journal')
plt.ylabel('Number of Retractions')
plt.xticks(rotation=45)
plt.savefig('Top_Journals_Bar_Chart.png')  # Saves the plot as a PNG file
plt.close()
