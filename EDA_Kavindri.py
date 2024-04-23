import pandas as pd
import matplotlib.pyplot as plt

# Load the dataset
data = pd.read_csv('C:/Users/kavi_/OneDrive/Desktop/CDU lectures/Semester 3/PRT 564/Python_workings/final_retractions35215.csv')

#finding a trend in subjects for retractions

#splitting subject column by semicolon delimiter and adding all subjects to the same column to get popularity
subject = data["Subject"].str.split(';').explode().value_counts()

#restting index to convert to dataframe
subject = subject.reset_index()
subject.columns = ['Subject', 'Count']
subject = subject.iloc[1:, :]

sub = subject.head(10)
nsub = subject.iloc[10:,:]
amount = nsub['Count'].sum()
print(amount)

#plotting subject against number of retractions
fig, ax = plt.subplots()
ax.pie(sub['Count'], labels=sub['Subject'], autopct='%1.1f%%')
plt.show()

#finding a trend in reasons for retractions column
#Splitting values in Reason column and removing '+' in beginning of each value
Reason = data["Reason"].str.split(';').apply(lambda x: [item.lstrip('+') for item in x]).explode().value_counts()

#converting series into a dataframe and renaming columns
Reason = Reason.reset_index()
Reason.columns = ['Reason', 'Count']
Reason = Reason.iloc[1:, :]

#plotting 
top10 = Reason.head(10)
fig, ax = plt.subplots()
ax.pie(top10['Count'], labels=top10['Reason'], autopct='%1.1f%%')
plt.show()


# # Bar chart of top 10 Countries by number of retractions
# plt.figure(figsize=(12, 8))
# data['Country'].value_counts().head(10).plot(kind='bar', color='blue')
# plt.title('Top Countries by Number of Retractions')
# plt.xlabel('Country')
# plt.ylabel('Number of Retractions')
# plt.xticks(rotation=45)
# plt.savefig('Top_Countries_Bar_Chart.png')  # Saves the plot as a PNG file
# plt.show()