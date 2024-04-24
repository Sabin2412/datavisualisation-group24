import pandas as pd
import matplotlib.pyplot as plt

# Load your dataset
data = pd.read_csv('cleaned_retractions35215.csv')

# Convert 'RetractionDate' to datetime format, if needed
data['RetractionDate'] = pd.to_datetime(data['RetractionDate'], errors='coerce')

# Splitting the 'Reason' column where reasons are separated by semicolons
reasons_expanded = data['Reason'].str.split(';', expand=True)

# Stack the DataFrame to convert it from wide to long format
reasons_long = reasons_expanded.stack().reset_index(level=1, drop=True).rename('ExpandedReason')

# Filter out missing or blank reasons before counting occurrences
reasons_long = reasons_long[reasons_long.str.strip().astype(bool)]

# Now, join the long format reasons back to the main data keeping the index to retain the reference to original rows
data_expanded_reasons = data.join(reasons_long, how='outer')

# Check if 'ExpandedReason' is in the columns and if it has non-empty values
if 'ExpandedReason' in data_expanded_reasons.columns and not data_expanded_reasons['ExpandedReason'].empty:
    # Group by the expanded reasons and count retractions
    reason_trend = data_expanded_reasons['ExpandedReason'].value_counts()

    # Get top reasons with the most retractions (adjust the number as needed)
    top_reasons = reason_trend.head(10)

    # Plotting the trend of retractions by reason
    plt.figure(figsize=(10, 8))
    top_reasons.plot(kind='bar')
    plt.title('Top Reasons for Retractions')
    plt.xlabel('Reason')
    plt.ylabel('Number of Retractions')
    plt.xticks(rotation=45)  # Rotate labels for better readability
    plt.tight_layout()  # Adjust layout to make room for label rotation
    plt.show()
else:
    print("ExpandedReason column not found or empty. Please check the dataset and column names.")
