import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.formula.api import ols

# Loading  our dataset
data = pd.read_csv('/Users/sabinghimire/data visualisation/final_retractions35215.csv')
data['RetractionDate'] = pd.to_datetime(data['RetractionDate'], errors='coerce')
data['Year'] = data['RetractionDate'].dt.year

# Aggregate data by year
yearly_counts = data['Year'].value_counts().sort_index()

# Preparing  DataFrame for regression
df = pd.DataFrame(yearly_counts).reset_index()
df.columns = ['Year', 'Retractions']

# Adding a constant term for the intercept
df['Intercept'] = 1

# Fitting a linear regression model
model = ols('Retractions ~ Year', data=df).fit()

# Summary of the model
print(model.summary())

# Plotting the trend along with the regression line
plt.figure(figsize=(10, 5))
plt.scatter(df['Year'], df['Retractions'], color='blue', label='Actual Retractions')
plt.plot(df['Year'], model.fittedvalues, color='red', label='Fitted Line')
plt.title('Retractions Trend Over Years')
plt.xlabel('Year')
plt.ylabel('Number of Retractions')
plt.legend()
plt.show()
