import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv('C:/Users/kavi_/OneDrive/Desktop/CDU lectures/Semester 3/PRT 564/Project/cleaned_retractions35215.csv')

#information on the data set
print(data.info())
#data set has no null values

#Finding outliers on the only quantitative variable - citation count 
print(data.describe()[['CitationCount']])

# Box plot
plt.boxplot(data['CitationCount'])
plt.show()
#box plot shows there are outliers 

#finding outliers statistically 
subset = data['CitationCount']
q1 = subset.quantile(0.25)
q3 = subset.quantile(0.75)
IQR = q3-q1
outliers = subset[((subset<(q1-(1.5*IQR)))|(subset>(q3+(1.5*IQR))))]

print("Number of Outliers - "+str(len(outliers)))
print("Maximum Outlier - "+str(outliers.max()))
print("Minimum Outlier - "+str(outliers.min()))

#we can remove outliers from the entire data set
data = data.drop(data[data['CitationCount'].isin(outliers)].index)
print(str(len(data)))

data.to_csv('final_retractions35215.csv', index=False)