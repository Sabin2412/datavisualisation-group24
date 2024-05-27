# import pandas as pd
# import matplotlib.pyplot as plt

# # Load the dataset
# data = pd.read_csv('C:/Users/kavi_/OneDrive/Desktop/CDU lectures/Semester 3/PRT 564/Python_workings/Final_Retractions.csv')

# #Finding a trend in Countries for retractions
# #Counts an article for a country if at least one co-author has an affiliation in that country.

# #splitting Country semicolon delimiter and getting retraction count
# Country = data["Country"].str.split(';').explode().value_counts().reset_index()
# Country.columns = ['Country', 'Count']

# #The top retracted countries
# # print(Country)
# sub = Country.head(5)

# #Creating a row value for the rest of the countries as others and their total count
# new_row = pd.DataFrame(data = {'Country' : ['Others'], 'Count' : [Country['Count'].iloc[5:].sum()]})
# sub = pd.concat([sub, new_row])

# #plotting Country against number of retractions as a pie chart
# fig, ax = plt.subplots()
# ax.pie(sub['Count'], labels=sub['Country'], autopct='%1.1f%%')
# plt.title("Top 5 Countries  and their retraction percentage")
# plt.legend(loc='upper left', fontsize='small', bbox_to_anchor=(1.05, 1))
# #plt.show()

# # Filter the dataframe for retractions from country
# top5 = Country.head(5)
# print(top5)
# yearly_retractions = pd.DataFrame()

# # Extract yearly retractions for each of the top 5 countries
# for cnt in top5["Country"]:
#     country_data = data[data['Country'].str.contains(cnt, na=False, case=False)]
#     yearly_data = country_data['RetractedYear'].value_counts().sort_index()
#     yearly_retractions[cnt] = yearly_data
# print(yearly_retractions)
# # Calculate total yearly retractions across all countries
# total_yearly_retractions = data['RetractedYear'].value_counts().sort_index()
# yearly_retractions['Total'] = total_yearly_retractions

# # Plotting the data
# plt.figure(figsize=(12, 8))
# for cnt in top5:
#     plt.plot(yearly_retractions.index, yearly_retractions[cnt], label=cnt)

# plt.plot(yearly_retractions.index, yearly_retractions['Total'], label='Total', linewidth=3, linestyle='--')

# plt.xlabel('Year')
# plt.ylabel('Number of Retractions')
# plt.title('Yearly Retractions for Top 5 Countries and Total Retractions')
# plt.legend()
# plt.grid(True)
# plt.show()
import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV file
file_path = 'C:/Users/kavi_/OneDrive/Desktop/CDU lectures/Semester 3/PRT 564/Python_workings/Final_Retractions.csv'
df = pd.read_csv(file_path)

# Filter out rows with missing 'RetractedYear' or 'Country'
df = df.dropna(subset=['RetractedYear', 'Country'])

# Expand the rows where multiple countries are listed
df['Country'] = df['Country'].str.split(';')
df = df.explode('Country')

# Trim whitespace around country names
df['Country'] = df['Country'].str.strip()

# Group by country and count the number of retractions
country_retraction_counts = df['Country'].value_counts()
print(country_retraction_counts)
# Identify the top 5 countries with the most retractions
top_5_countries = country_retraction_counts.head(5).index

# Initialize a dataframe to hold yearly retractions data
yearly_retractions = pd.DataFrame()

# Extract yearly retractions for each of the top 5 countries
for country in top_5_countries:
    country_data = df[df['Country'] == country]
    yearly_data = country_data['RetractedYear'].value_counts().sort_index()
    yearly_retractions[country] = yearly_data

# Calculate total yearly retractions across all countries
total_yearly_retractions = df['RetractedYear'].value_counts().sort_index()
yearly_retractions['Total'] = total_yearly_retractions

# Fill NaN values with 0 for plotting
yearly_retractions = yearly_retractions.fillna(0)

# Plotting the data
plt.figure(figsize=(14, 10))

# Plot for each of the top 5 countries
for country in top_5_countries:
    plt.plot(yearly_retractions.index, yearly_retractions[country], label=country)

# Plot the total retractions
plt.plot(yearly_retractions.index, yearly_retractions['Total'], label='Total', linewidth=3, linestyle='--', color='black')

plt.xlabel('Year')
plt.ylabel('Number of Retractions')
plt.title('Yearly Retractions for Top 5 Countries and Total Retractions')
plt.legend()
plt.grid(True)
plt.show()