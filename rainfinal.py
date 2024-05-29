import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.utils.class_weight import compute_class_weight
from sklearn.preprocessing import KBinsDiscretizer
import numpy as np

# Load the dataset
df = pd.read_csv('finalred.csv')

# Check for missing values and drop them if any
df.dropna(inplace=True)

# Identify the target variable and feature columns
target = 'CitationCount'
feature_columns = ['RetractedYear', 'OriginalPaperYear', 'SVD_Component_1', 'SVD_Component_2', 'SVD_Component_3']

# Bin the CitationCount into discrete classes
bins = KBinsDiscretizer(n_bins=5, encode='ordinal', strategy='uniform')
df['CitationCountBinned'] = bins.fit_transform(df[[target]]).astype(int)

# Define a function to evaluate a model
def evaluate_model(model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='macro')
    recall = recall_score(y_test, y_pred, average='macro')
    f1 = f1_score(y_test, y_pred, average='macro')
    return accuracy, precision, recall, f1

# Split the data into training and testing sets
X = df[feature_columns]
y = df['CitationCountBinned']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the models
logistic_regression = LogisticRegression(random_state=42, max_iter=1000)
naive_bayes = GaussianNB()
random_forest = RandomForestClassifier(n_estimators=100, random_state=42)

# Evaluate the models
models = {
    'Logistic Regression': logistic_regression,
    'Naïve Bayes': naive_bayes,
    'Random Forest': random_forest
}

results = {}
for model_name, model in models.items():
    accuracy, precision, recall, f1 = evaluate_model(model, X_train, X_test, y_train, y_test)
    results[model_name] = {
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1 Score': f1
    }

# Display the results
results_df = pd.DataFrame(results).T
print("Model Evaluation Results:")
print(results_df)

# Compute class weights
class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
class_weights_dict = {i: class_weights[i] for i in range(len(class_weights))}

# Initialize the Random Forest model with class weights
rf = RandomForestClassifier(random_state=42, class_weight=class_weights_dict)

# Define the parameter grid
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'bootstrap': [True, False]
}

# Initialize Stratified K-Fold
stratified_kfold = StratifiedKFold(n_splits=3)

# Perform Grid Search with stratified cross-validation
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=stratified_kfold, n_jobs=-1, verbose=2)
grid_search.fit(X_train, y_train)

# Get the best parameters and the best model
best_params = grid_search.best_params_
best_rf_model = grid_search.best_estimator_

# Evaluate the best model
y_pred_best_rf = best_rf_model.predict(X_test)
accuracy_best_rf = accuracy_score(y_test, y_pred_best_rf)
precision_best_rf = precision_score(y_test, y_pred_best_rf, average='macro')
recall_best_rf = recall_score(y_test, y_pred_best_rf, average='macro')
f1_best_rf = f1_score(y_test, y_pred_best_rf, average='macro')

print("\nBest Random Forest Model - Evaluation Metrics:")
print(f"Accuracy: {accuracy_best_rf}")
print(f"Precision: {precision_best_rf}")
print(f"Recall: {recall_best_rf}")
print(f"F1 Score: {f1_best_rf}")
print(f"Best Parameters: {best_params}")
