import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Load your dataset
csv_file_path = '/Users/sabinghimire/data visualisation/clean csv/cleaned_retractions35215.csv'
df = pd.read_csv(csv_file_path)

# Clean and prepare the ArticleType column
df['ArticleType'] = df['ArticleType'].apply(lambda x: x.strip(';').split(';'))

# Explode the DataFrame so each article type gets its own row
exploded_df = df.explode('ArticleType')

# Count occurrences of each article type and get the top 5
top_article_types = exploded_df['ArticleType'].value_counts().head(5)

# Calculate percentages
total = top_article_types.sum()
percentages = (top_article_types / total * 100).round(1)

# Create a donut chart
fig, ax = plt.subplots(figsize=(8, 8))
wedges, _ = ax.pie(top_article_types, labels=None, startangle=90, wedgeprops={'width': 0.3, 'edgecolor': 'w'})

# Add custom legend handles with percentages
legend_handles = []
for wedge, label, percentage in zip(wedges, top_article_types.index, percentages):
    handle = mpatches.Patch(label=f"{label}: {percentage}%", color=wedge.get_facecolor())
    legend_handles.append(handle)

# Place legend box on the side with custom handles
plt.legend(handles=legend_handles, title="Article Types", loc="center left", bbox_to_anchor=(1, 0.5))

# Set aspect ratio to be equal to ensure that pie is drawn as a circle
ax.set_aspect('equal')

plt.title('Distribution of Top 5 Article Types')
plt.show()
