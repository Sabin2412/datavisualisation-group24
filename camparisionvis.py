import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Assume 'df' is your DataFrame from the output of the feature selection comparison
data = {
    'Feature': [
        'Record ID', 'RetractionDOI', 'OriginalPaperDOI', 'RetractionDate', 'Journal',
        'OriginalPaperDate', 'Reason', 'Notes', 'Subject', 'RetractionPubMedID',
        'Publisher', 'URLS', 'Institution', 'OriginalPaperPubMedID', 'Title',
        'ArticleType', 'Author', 'Country', 'CitationCount', 'RetractionNature'
    ],
    'Mutual_Info': [
        0.087539, 0.065804, 0.060529, 0.056468, 0.055477,
        0.043854, 0.037391, 0.029859, 0.023634, 0.021843,
        0.018006, 0.016014, 0.014583, 0.010780, 0.010472,
        0.008333, 0.007037, 0.004601, 0.001601, 0.001075
    ],
    'Chi_Squared': [
        16177720, 337597.2, 328799.9, 1000353, 190689.5,
        797053.3, 245760.3, 29709, 45515.32, 3299019000,
        322.7447, 123781.2, 80133.95, 2887665000, 157183,
        636.585, 25544.12, 7591.773, 932.8353, np.nan
    ],
    'Correlation': [
        1, 1, 0, 0, 1,
        0, 1, 1, 1, 1,
        1, 1, 1, 0, 1,
        1, 1, 1, 1, 1
    ],
    'MI_Rank': [
        1, 2, 3, 4, 5,
        6, 7, 8, 9, 10,
        11, 12, 13, 14, 15,
        16, 17, 18, 19, 20
    ],
    'Chi_Rank': [
        3, 6, 7, 4, 9,
        5, 8, 14, 13, 1,
        19, 11, 12, 2, 10,
        18, 15, 16, 17, np.nan
    ],
    'Corr_Rank': [
        1, 1, 17, 17, 1,
        17, 1, 1, 1, 1,
        1, 1, 1, 17, 1,
        1, 1, 1, 1, 1
    ]
}

df = pd.DataFrame(data)
df.set_index('Feature', inplace=True)

# Plotting
plt.figure(figsize=(14, 10))
sns.heatmap(df, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
plt.title('Feature Selection Comparison')
plt.show()
