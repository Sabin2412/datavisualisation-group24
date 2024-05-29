import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load the data
df = pd.read_csv('/Users/sabinghimire/data visualisation/clean csv/cleaned_retractions35215.csv')

# Fill missing values if any, using forward fill method
df.ffill(inplace=True)

# Initialize label encoder
le = LabelEncoder()

# Encoding categorical variables including the target
categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
for col in categorical_cols:
    df[col] = le.fit_transform(df[col])

# Select features, excluding known identifiers, textual content, and the target variable
feature_cols = [col for col in df.columns if col not in ['Record ID', 'Title', 'URLS', 'Notes', 'OriginalPaperDOI', 'RetractionDOI', 'RetractionDate', 'OriginalPaperDate', 'Paywalled']]

# Prepare feature matrix X and target vector y
X = df[feature_cols]
y = df['Paywalled']

# Feature selection using Mutual Information
selector = SelectKBest(mutual_info_classif, k=10)
X_selected = selector.fit_transform(X, y)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.3, random_state=42)

# Train a RandomForest classifier
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# Predict on the test set and evaluate
predictions = rf.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
print(f"Accuracy of the RandomForest model: {accuracy}")

# Output selected features correctly aligning scores with the selected features
scores = selector.scores_[selector.get_support()]
selected_features = pd.Series(scores, index=X.columns[selector.get_support()]).nlargest(10)
print("Top 10 selected features based on Mutual Information:\n", selected_features)
