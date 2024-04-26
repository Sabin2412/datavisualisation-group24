import pandas as pd
import matplotlib.pyplot as plt

# Load your dataset
data = pd.read_csv('/Users/sabinghimire/data visualisation/final_retractions35215.csv')

# Convert 'RetractionDate' to datetime format
data['RetractionDate'] = pd.to_datetime(data['RetractionDate'], errors='coerce')

# Splitting the 'Country' column where countries are separated by semicolons
countries_expanded = data['Country'].str.split(';', expand=True)

# Stack the DataFrame to convert it from wide to long format
countries_long = countries_expanded.stack().reset_index(level=1, drop=True).rename('ExpandedCountry')

# Join the long format countries back to the main data keeping the index to retain the reference to original rows
data_expanded_countries = data.join(countries_long, how='outer')

# Now 'ExpandedCountry' is our target column for counting
if 'ExpandedCountry' in data_expanded_countries.columns:
    # Group by the expanded countries and count retractions
    country_trend = data_expanded_countries['ExpandedCountry'].value_counts()

    # Get top 10 countries with the most retractions
    top_10_countries = country_trend.head(10)

    # Plotting the trend of retractions by top 10 countries
    plt.figure(figsize=(10, 8))
    top_10_countries.plot(kind='bar')
    plt.title('Top 10 Countries by Number of Retractions')
    plt.xlabel('Country')
    plt.ylabel('Number of Retractions')
    plt.xticks(rotation=45)  # Rotate labels for better readability
    plt.tight_layout()  # Adjust layout to make room for label rotation
    plt.show()
else:
    print("ExpandedCountry column not found. Please check the dataset and column names.")
