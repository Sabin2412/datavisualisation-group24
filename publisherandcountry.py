import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Load your dataset
csv_file_path = 'clean csv/reduced.csv'  # Make sure to use the correct path to your CSV file
df = pd.read_csv(csv_file_path)

# Prepare data - split countries and explode if there are multiple countries per row
df['Country'] = df['Country'].str.split(';').apply(lambda x: [country.strip() for country in x])
exploded_df = df.explode('Country')

# Group data by country and count retractions
country_retractions = exploded_df.groupby('Country').size().reset_index(name='RetractionCount')

# Identify top 5 countries by total retractions
top_countries = country_retractions.nlargest(5, 'RetractionCount')

# Load a world map
world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))

# Ensure that 'Country' in top_countries is in a format that matches 'name' in the world dataframe
top_countries['Country'] = top_countries['Country'].str.title()  # Adjust for proper capitalization if necessary

# Merge the top countries data with the world map data
world_with_data = world.merge(top_countries, how="left", left_on="name", right_on='Country')

# Plot
fig, ax = plt.subplots(1, 1, figsize=(15, 10))
world_with_data.boundary.plot(ax=ax)

# Plotting countries with their retraction count
# Setting colors for top 5 countries
cmap = plt.cm.get_cmap('viridis', 5)  # Get a color map with 5 distinct colors
plot = world_with_data.dropna().plot(column='RetractionCount', ax=ax, legend=False, cmap=cmap)

# Fill countries with no data as 'lightgrey'
world_with_data[world_with_data['RetractionCount'].isna()].plot(ax=ax, color='lightgrey')

# Creating a custom legend
# Generate a label and color patch for each top country
legend_labels = [f'{country}: {int(count)} Retractions' for country, count in zip(top_countries['Country'], top_countries['RetractionCount'])]
legend_patches = [mpatches.Patch(color=cmap(i), label=label) for i, label in enumerate(legend_labels)]

# Add the custom legend to the plot
ax.legend(handles=legend_patches, title="Top 5 Countries by Retractions", loc='upper left')

plt.title('Top 5 Countries by Number of Retractions')
plt.show()
