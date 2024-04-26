import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Specify the path to your CSV file
csv_file_path = '/Users/sabinghimire/data visualisation/clean csv/reduced.csv'

# Load your dataset
df = pd.read_csv(csv_file_path)

# Select only the columns of interest
data = df[['CitationCount', 'RetractedYear', 'OriginalPaperYear']]

# Calculate the correlation matrix
correlation_matrix = data.corr()

# Create a heatmap using seaborn
plt.figure(figsize=(8, 6))  # Adjust the size of the plot as needed
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=.5, square=True)

# Add title
plt.title('Correlation Heatmap')

# Show the plot
plt.show()
