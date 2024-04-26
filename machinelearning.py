import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

# Load the dataset
data = pd.read_csv('/Users/sabinghimire/data visualisation/clean csv/cleaned_retractions35215.csv')

# Convert 'RetractionDate' and 'OriginalPaperDate' to datetime and create a feature for the time to retraction
data['RetractionDate'] = pd.to_datetime(data['RetractionDate'])
data['OriginalPaperDate'] = pd.to_datetime(data['OriginalPaperDate'])
data['TimeToRetraction'] = (data['RetractionDate'] - data['OriginalPaperDate']).dt.days

# Selecting features and target variable
# 'Subject' and 'Journal' could be categorical features and 'TimeToRetraction' is numerical
# Assuming 'Paywalled' is a binary variable, where 'Yes' means it is paywalled and 'No' means it is not
features = data[['TimeToRetraction', 'Subject', 'Journal', 'Paywalled']]
target = data['CitationCount']  # Target variable

# Handling categorical variables with one-hot encoding and missing values
categorical_features = ['Subject', 'Journal', 'Paywalled']
numerical_features = ['TimeToRetraction']

# Preprocessing for categorical data
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))])

# Preprocessing for numerical data
numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())])

# Bundle preprocessing for numerical and categorical data
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)])

# Split the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

# Create and train the model
model = Pipeline(steps=[('preprocessor', preprocessor),
                        ('regressor', LinearRegression())])

model.fit(X_train, y_train)

# Predicting on the test set
y_pred = model.predict(X_test)

# Evaluating the model
mse = mean_squared_error(y_test, y_pred)
r_squared = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse}")
print(f"R-squared: {r_squared}")


# Access the regressor to get coefficients (if you want to interpret the coefficients)
# Note: This will not work directly due to one-hot encoding. You will need to map the encoded features back to their original names.
