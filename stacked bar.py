import pandas as pd
import matplotlib.pyplot as plt

# Load the dataset
data = pd.read_csv('C:/Users/kavi_/OneDrive/Desktop/CDU lectures/Semester 3/PRT 564/Python_workings/Final_Retractions.csv')

#splitting Article Type semicolon delimiter and getting retraction count
art = data["ArticleType"].str.rstrip('; ').str.split(';')

artsub = art.explode().value_counts().reset_index()
artsub.columns = ['ArticleType', 'Count']
artsub = art.head(2)

df = pd.concat([data['RetractedYear'],art], axis=1).explode('ArticleType')
counts = df.groupby(['RetractedYear', 'ArticleType']).value_counts().explode().reset_index()
counts = counts.sort_values(by='count', ascending=False)

counts['RetractedYear'] = counts['RetractedYear'].astype(int)
counts = counts[counts['RetractedYear']>=2005]
counts['RetractedYear'] = counts['RetractedYear'].astype(str)

# Filter the DataFrame for 'Research Article' and 'Conference Abstract/Paper' article types
filtered_df = counts[counts['ArticleType'].isin(['Research Article', 'Conference Abstract/Paper'])]

# Pivot the DataFrame to prepare for plotting
pivot_df = filtered_df.pivot(index='RetractedYear', columns='ArticleType', values='count')

# Plot the stacked bar chart
pivot_df.plot(kind='bar', stacked=True, figsize=(10, 6), color=['Green', 'Blue'])

# Add labels and title
plt.xlabel('Retracted Year')
plt.ylabel('Retraction Count')
plt.title('Stacked Bar Chart of Top 2 Retracted Article Types per Year')

# Show the plot
plt.legend(title='Article Type')
plt.show()