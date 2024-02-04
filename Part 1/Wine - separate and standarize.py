import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

# Load dataset
df = pd.read_csv('A1-wine/winequality-white.csv', sep=';')

# Check if there's any missing data
if df.isnull().values.any():
    # Replace null values with 0
    df.fillna(0, inplace=True)

# Standardize the dataset
scaler = preprocessing.StandardScaler().fit(df)
df_scaled = scaler.transform(df)
df_scaled = pd.DataFrame(df_scaled, columns=df.columns)

# Save them in two .txt files, using tabulation as a marker
df_scaled.to_csv('A1-wine/wine-data.txt', sep='\t', index=False)
