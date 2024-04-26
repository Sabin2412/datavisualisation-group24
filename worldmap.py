import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt

# Load your dataset
csv_file_path = '/Users/sabinghimire/data visualisation/clean csv/cleaned_retractions35215.csv'
df = pd.read_csv(csv_file_path)

# Prepare data
df['Retracted'] = df['RetractionNature'].notna()
df['Authors'] = df['Author'].str.split(';').apply(lambda x: [author.strip() for author in x])
exploded_df = df.explode('Authors')

# Group data by author and country, and count retractions
author_country_retractions = exploded_df.groupby(['Authors', 'Country'])['Retracted'].sum().reset_index()

# Identify top 10 authors by total retractions
top_authors = author_country_retractions.groupby('Authors')['Retracted'].sum().nlargest(10).index

# Filter data to include only top 10 authors
top_authors_data = author_country_retractions[author_country_retractions['Authors'].isin(top_authors)]

# Load a world map
world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))

# Merge the top authors data with the world map data
world = world.merge(top_authors_data, how="left", left_on="name", right_on="Country")

# Plot
fig, ax = plt.subplots(1, 1, figsize=(15, 10))
world.boundary.plot(ax=ax)
world.plot(column='Retracted', ax=ax, legend=True,
           legend_kwds={'label': "Number of Retractions by Country"},
           cmap='OrRd', missing_kwds={'color': 'lightgrey'})

plt.title('World Heat Map of Retractions for Top 10 Authors')
plt.show()
