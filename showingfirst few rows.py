import pandas as pd

# Specify the paths
input_path = '/Users/sabinghimire/data visualisation/cleaned_retractions35215.csv'
output_path = '/Users/sabinghimire/data visualisation/first10rows.csv'

# Load the dataset
data = pd.read_csv(input_path)

# Display the first few rows
print(data.head())

# Save the output CSV file
data.to_csv(output_path, index=False)

print(f"Output saved to: {output_path}")
