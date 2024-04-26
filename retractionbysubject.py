import pandas as pd
import matplotlib.pyplot as plt

# Load your dataset
data = pd.read_csv('/Users/sabinghimire/data visualisation/clean csv/cleaned_retractions35215.csv')

# Remove any trailing semicolons from the 'Subject' column
data['Subject'] = data['Subject'].str.rstrip(';')

# Convert 'RetractionDate' to datetime format
data['RetractionDate'] = pd.to_datetime(data['RetractionDate'], errors='coerce')

# Splitting the 'Subject' column where subjects are separated by semicolons
subjects_expanded = data['Subject'].str.split(';', expand=True)

# Stack the DataFrame to convert it from wide to long format
subjects_long = subjects_expanded.stack().reset_index(level=1, drop=True).rename('ExpandedSubject')

# Clean the 'ExpandedSubject' strings (remove any whitespace)
subjects_long = subjects_long.str.strip()

# Join the long format subjects back to the main data keeping the index to retain the reference to original rows
data_expanded_subjects = data.join(subjects_long, how='outer')

# Group by the cleaned subjects and count retractions
subject_trend = data_expanded_subjects['ExpandedSubject'].value_counts()

# Get top 10 subjects with the most retractions
top_10_subjects = subject_trend.head(10)

# Plotting the trend of retractions by top 10 subjects
plt.figure(figsize=(10, 8))
top_10_subjects.plot(kind='bar')
plt.title('Top 10 Subjects by Number of Retractions')
plt.xlabel('Subject')
plt.ylabel('Number of Retractions')
plt.xticks(rotation=45)  # Rotate labels for better readability
plt.tight_layout()  # Adjust layout to make room for label rotation
plt.show()
