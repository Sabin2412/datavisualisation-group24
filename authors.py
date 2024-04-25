import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
data = pd.read_csv('C:/Users/kavi_/OneDrive/Desktop/CDU lectures/Semester 3/PRT 564/Python_workings/Final_Retractions.csv')

art = data["Author"].str.rstrip('; ').str.split(';').explode().value_counts().reset_index()
art.columns = ['Author', 'Count']

#getting the percentage of retractions per each author 
art['Percentage'] = (art['Count'] / art['Count'].sum()) * 100
print(art)

#getting the top 10 authors
art = art.head(10)

#plotting authors against number of retractions
fig, ax = plt.subplots()
patches, texts, _ = ax.pie(art['Percentage'], colors=sns.color_palette('Set3'), startangle=90, autopct='%1.1f%%')
labels = [f'{author}: {count}' for author, count in zip(art['Author'], art['Count'])]
plt.legend(patches, labels, loc='upper left', fontsize='small', bbox_to_anchor=(1.05, 1))
plt.title("Top Authors whose papers were retracted")
plt.show()