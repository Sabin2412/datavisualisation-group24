import pandas as pd
import matplotlib.pyplot as plt

# Load the dataset
data = pd.read_csv('C:/Users/kavi_/OneDrive/Desktop/CDU lectures/Semester 3/PRT 564/Project/cleaned_retractions35215.csv')

#finding a trend in subjects for retractions

#splitting subject column by semicolon delimiter adn removing extra characters
subject = data["Subject"].str.rstrip('; ').str.split(';')

#the top 5 most retracted subjects all time
sub = subject.explode().value_counts().reset_index()
sub.columns = ['Subject', 'Count']
sub= sub['Subject'].head(5)

#taking only year from the retraction date
date = data['RetractionDate'].apply(lambda x: x.rstrip('/').split('/')[-1])

#combining the two dataframes 
df = pd.concat([date,subject], axis=1).explode('Subject')

#getting the grouped count of Retraction Year and subject name
counts = df.groupby(['RetractionDate', 'Subject']).value_counts().explode().reset_index()

#getting only values after 2005 
counts['RetractionDate'] = counts['RetractionDate'].astype(int)
counts = counts[counts['RetractionDate']>=2005]
counts['RetractionDate'] = counts['RetractionDate'].astype(str)

#getting only the values of the top 5 most retracted subjects 
counts = counts[counts['Subject'].isin(sub)]

#pivoting the dataframe to plot 
pivot_counts = counts.pivot(index='RetractionDate', columns='Subject', values='count').fillna(0)

#plotting the line graph
pivot_counts.plot(kind='line', marker = 'o')
plt.xlabel('Year')
plt.ylabel('Retraction Count')
plt.title('Count of Retracted Subjects Over Time')
plt.legend(title='Subject')
plt.grid(True)
plt.show()


