import pandas as pd
import matplotlib.pyplot as plt

# Load the dataset
data = pd.read_csv('C:/Users/kavi_/OneDrive/Desktop/CDU lectures/Semester 3/PRT 564/Project/cleaned_retractions35215.csv')

#Finding a trend in Countries for retractions
#Counts an article for a country if at least one co-author has an affiliation in that country.

#splitting Country semicolon delimiter and getting retraction count
Country = data["Country"].str.split(';').explode().value_counts().reset_index()
Country.columns = ['Country', 'Count']

#The top retracted countries
sub = Country.head(5)

#Creating a row value for the rest of the countries as others and their total count
new_row = pd.DataFrame(data = {'Country' : ['Others'], 'Count' : [Country['Count'].iloc[5:].sum()]})
sub = pd.concat([sub, new_row])

#plotting Country against number of retractions as a pie chart
fig, ax = plt.subplots()
ax.pie(sub['Count'], labels=sub['Country'], autopct='%1.1f%%')
plt.show()


#splitting Article Type semicolon delimiter and getting retraction count
art = data["ArticleType"].str.rstrip('; ').str.split(';').explode().value_counts().reset_index()
art.columns = ['ArticleType', 'Count']
print(art)

artsub = art.head(5)
#Creating a row value for the rest of the Article types as others and their total count
new_row1 = pd.DataFrame(data = {'ArticleType' : ['Others'], 'Count' : [art['Count'].iloc[5:].sum()]})
artsub = pd.concat([artsub, new_row1])

#plotting Country against number of retractions
fig, ax = plt.subplots()
ax.pie(artsub['Count'], labels=artsub['ArticleType'], autopct='%1.1f%%')
plt.show()