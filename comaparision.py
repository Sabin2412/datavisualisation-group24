import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import SelectKBest, mutual_info_classif, chi2
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt

# Load the data
df = pd.read_csv('/Users/sabinghimire/data visualisation/clean csv/cleaned_retractions35215.csv')

# Fill missing values, using forward fill method
df.ffill(inplace=True)

# Initialize label encoder
le = LabelEncoder()

# Encoding categorical variables
categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
for col in categorical_cols:
    df[col] = le.fit_transform(df[col])

# Prepare feature matrix X and target vector y
X = df.drop('Paywalled', axis=1)  # Assuming 'Paywalled' is the target
y = df['Paywalled']

# Define feature selection functions
def get_feature_scores(X, y, score_func, method_name):
    selector = SelectKBest(score_func, k='all')
    selector.fit(X, y)
    scores = selector.scores_
    return pd.Series(scores, index=X.columns, name=method_name)

# Mutual Information Scores
mi_scores = get_feature_scores(X, y, mutual_info_classif, 'Mutual_Info')

# Chi-Squared Scores
X_positive = X - X.min()  # Ensure all values are non-negative
chi_scores = get_feature_scores(X_positive, y, chi2, 'Chi_Squared')

# Correlation-based Feature Selection
corr_matrix = X.corr().abs()
upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
to_drop = [column for column in upper_tri.columns if any(upper_tri[column] > 0.8)]
corr_features = pd.Series([1 if i not in to_drop else 0 for i in X.columns], index=X.columns, name='Correlation')

# Combine all scores into a single DataFrame
feature_scores = pd.concat([mi_scores, chi_scores, corr_features], axis=1)

# Rank the features within each method
feature_scores['MI_Rank'] = feature_scores['Mutual_Info'].rank(method='min', ascending=False)
feature_scores['Chi_Rank'] = feature_scores['Chi_Squared'].rank(method='min', ascending=False)
feature_scores['Corr_Rank'] = feature_scores['Correlation'].rank(method='min', ascending=False, na_option='bottom')

# Sort by MI Rank as an example
feature_scores = feature_scores.sort_values(by='MI_Rank')

# Display or save the table
print(feature_scores)
# feature_scores.to_csv('/path/to/feature_scores_comparison.csv')




# Define a function to create a heatmap for feature scores and rankings
def plot_feature_selection_scores(feature_scores):
    # Normalize the feature scores for better visualization (optional)
    normalized_scores = feature_scores.copy()
    normalized_scores['Mutual_Info'] = normalized_scores['Mutual_Info'] / normalized_scores['Mutual_Info'].max()
    normalized_scores['Chi_Squared'] = normalized_scores['Chi_Squared'] / normalized_scores['Chi_Squared'].max()

    # Remove binary correlation features for heatmap scale compatibility
    feature_ranks = normalized_scores[['MI_Rank', 'Chi_Rank', 'Corr_Rank']]
    
    # Plotting
    plt.figure(figsize=(12, 10))
    sns.heatmap(feature_ranks, annot=True, cmap='coolwarm', fmt=".1f", linewidths=.5, cbar_kws={'label': 'Ranking Scale'})
    plt.title('Feature Selection Ranking Comparison')
    plt.ylabel('Feature')
    plt.xlabel('Feature Selection Method')
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.show()

# Call the visualization function
plot_feature_selection_scores(feature_scores)

#for comapring feature scores

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB

# Logistic Regression
lr = LogisticRegression(max_iter=1000)
lr.fit(X, y)
lr_importances = np.abs(lr.coef_[0])  # Taking the absolute values of coefficients

# Naive Bayes
nb = GaussianNB()
nb.fit(X, y)
nb_importances = np.abs(nb.theta_[1] - nb.theta_[0])  # Difference between means of each feature across classes

# Random Forest
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X, y)
rf_importances = rf.feature_importances_

# Create a DataFrame for the feature importances
importances_df = pd.DataFrame({
    'Features': X.columns,
    'Logistic Regression': lr_importances,
    'Naive Bayes': nb_importances,
    'Random Forest': rf_importances
})

# Normalize the importances
importances_df['Logistic Regression'] /= importances_df['Logistic Regression'].max()
importances_df['Naive Bayes'] /= importances_df['Naive Bayes'].max()
importances_df['Random Forest'] /= importances_df['Random Forest'].max()

# Rank the importances
importances_df['LR Rank'] = importances_df['Logistic Regression'].rank(ascending=False)
importances_df['NB Rank'] = importances_df['Naive Bayes'].rank(ascending=False)
importances_df['RF Rank'] = importances_df['Random Forest'].rank(ascending=False)

# Sort the DataFrame by one of the ranks for better visualization
importances_df.sort_values('LR Rank', inplace=True)

print("Normalized Feature Importances:")
print(importances_df[['Features', 'Logistic Regression', 'Naive Bayes', 'Random Forest']].to_string(index=False))
print("\nFeature Importance Rankings:")
print(importances_df[['Features', 'LR Rank', 'NB Rank', 'RF Rank']].to_string(index=False))



plt.figure(figsize=(15, 10))
sns.heatmap(importances_df.set_index('Features')[['LR Rank', 'NB Rank', 'RF Rank']],
            annot=True, fmt=".0f", cmap='coolwarm', linewidths=.5)
plt.title('Feature Importance Ranking Comparison Across Models')
plt.show()

