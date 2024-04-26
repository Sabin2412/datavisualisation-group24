import matplotlib
matplotlib.use('TkAgg')  # Adjust as per your environment capabilities

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import numpy as np

# Load the dataset
data = pd.read_csv('/Users/sabinghimire/data visualisation/clean csv/cleaned_retractions35215.csv')

# Ensure 'RetractionDate' is a datetime type
data['RetractionDate'] = pd.to_datetime(data['RetractionDate'])

# Extract year and group by it
data['Year'] = data['RetractionDate'].dt.year
yearly_counts = data.groupby('Year').size().reset_index(name='Counts')

# Filter data from the year 2000 onwards
yearly_counts = yearly_counts[yearly_counts['Year'] >= 2000]

# Prepare data for regression
X = yearly_counts['Year'].values.reshape(-1, 1)  # Reshape for scikit-learn
y = yearly_counts['Counts'].values

# Create and fit the model
model = LinearRegression()
model.fit(X, y)

# Make predictions
yearly_counts['Predicted'] = model.predict(X)

# Compute metrics
y_pred = model.predict(X)
r2 = r2_score(y, y_pred)
adjusted_r2 = 1 - (1 - r2) * (len(y) - 1) / (len(y) - X.shape[1] - 1)
mse = mean_squared_error(y, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y, y_pred)

# Plotting the results
plt.figure(figsize=(10, 6))
plt.scatter(yearly_counts['Year'], yearly_counts['Counts'], color='blue', label='Actual Retractions')
plt.plot(yearly_counts['Year'], yearly_counts['Predicted'], color='red', linewidth=2, label='Regression Line')
plt.title('Trend of Retractions Over Time (From 2000 onwards)')
plt.xlabel('Year')
plt.ylabel('Number of Retractions')
plt.legend()
plt.grid(True)

# Display metrics
plt.text(0.05, 0.6, f'R-squared: {r2:.3f}\nAdjusted R-squared: {adjusted_r2:.3f}\nMSE: {mse:.3f}\nRMSE: {rmse:.3f}\nMAE: {mae:.3f}',
         transform=plt.gca().transAxes, fontsize=12, verticalalignment='top')

plt.show()
