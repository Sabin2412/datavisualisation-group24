import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib
import matplotlib.pyplot as plt
import warnings

# Suppress warnings for least populated class
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn.model_selection._split')

# Load the updated reduced dataset with CitationCount
df = pd.read_csv('finalpca.csv')

# Check for missing values and drop them if any
df.dropna(inplace=True)

# Remove classes with less than 2 instances
class_counts = df['CitationCount'].value_counts()
valid_classes = class_counts[class_counts > 1].index
df = df[df['CitationCount'].isin(valid_classes)]

# Identify the target variable and feature columns
target = 'CitationCount'
feature_columns = [
    'PC1', 'PC2', 'PC3', 'PC4', 'PC5', 
    'PC6', 'PC7', 'PC8', 'PC9', 'PC10', 
    'PC11', 'PC12', 'PC13', 'PC14', 'PC15'
]

# Define a function to evaluate a model
def evaluate_model(model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='macro', zero_division=1)
    recall = recall_score(y_test, y_pred, average='macro', zero_division=1)
    f1 = f1_score(y_test, y_pred, average='macro', zero_division=1)
    return accuracy, precision, recall, f1

# Split the data into training and testing sets
X = df[feature_columns]
y = df[target]
splitter = StratifiedShuffleSplit(n_splits=2, test_size=0.2, random_state=42)
for train_index, test_index in splitter.split(X, y):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

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
best_f1_score = 0
best_model = None

for model_name, model in models.items():
    accuracy, precision, recall, f1 = evaluate_model(model, X_train, X_test, y_train, y_test)
    results[model_name] = {
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1 Score': f1
    }
    # Check if this model has the best F1 score
    if f1 > best_f1_score:
        best_f1_score = f1
        best_model = model

# Display the results
results_df = pd.DataFrame(results).T
print(results_df)

# Save the best model
if best_model:
    joblib.dump(best_model, 'finalbest_model.pkl')
    print(f"Best model saved as 'best_model.pkl' with F1 Score: {best_f1_score}")

# Visualize the performance of models
results_df.plot(kind='bar', figsize=(14, 8))
plt.title('Model Performance Comparison')
plt.ylabel('Score')
plt.xlabel('Model')
plt.xticks(rotation=0)
plt.legend(loc='lower right')
plt.tight_layout()
plt.savefig('finalmodel_performance_comparison.png')  # Save the plot
plt.show()
