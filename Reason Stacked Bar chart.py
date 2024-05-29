import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV file
file_path = 'C:/Users/kavi_/OneDrive/Desktop/CDU lectures/Semester 3/PRT 564/Python_workings/Final_Retractions.csv'
df = pd.read_csv(file_path)

# Split reasons and explode them into separate rows
retractions_exploded = df.assign(Reason=df['Reason'].str.split(';')).explode('Reason')
# print(retractions_exploded)
# Strip whitespace from reasons
retractions_exploded['Reason'] = retractions_exploded['Reason'].str.strip()

# Count occurrences of each reason
reason_counts = retractions_exploded['Reason'].value_counts()
reason_counts = reason_counts.drop(reason_counts.index[0])

print(reason_counts)
# Identify the top 3 reasons
top_3_reasons = reason_counts.head(5).index.tolist()
print(top_3_reasons)
# Categorize reasons into top 5 and 'Others'
retractions_exploded['Reason Category'] = retractions_exploded['Reason'].apply(lambda x: x if x in top_3_reasons else 'Others')

# Filter data from 2005 onwards
retractions_filtered = retractions_exploded[retractions_exploded['RetractedYear'] >= 2005]
print(retractions_filtered)
# Create a pivot table for the stacked bar chart
pivot_table = retractions_filtered.pivot_table(index='RetractedYear', columns='Reason Category', aggfunc='size', fill_value=0)

# Normalize the pivot table to convert counts to percentages
pivot_table_percentage = pivot_table.div(pivot_table.sum(axis=1), axis=0) * 100

# Plot the percentage stacked bar chart
pivot_table_percentage.plot(kind='bar', stacked=True, figsize=(12, 8), colormap='Set1')

plt.title('Top 5 Retraction Reasons Over Years (from 2005) as Percentages')
plt.xlabel('Year')
plt.ylabel('Percentage of Retractions')
plt.legend(title='Retraction Reason', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.xticks(rotation=45)
plt.tight_layout()

# Display the plot
plt.show()
