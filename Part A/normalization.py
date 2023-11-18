# Import necessary libraries
import pandas as pd

# Load the datasets
dataNorm = pd.read_csv('A1-synthetic.txt', sep='\t')
dataStand = pd.read_csv('A1-turbine.txt', sep='\t')

# Normalization
for column in dataNorm.columns:
    dataNorm[column] = (dataNorm[column] - dataNorm[column].min()) / (dataNorm[column].max() - dataNorm[column].min())
dataNorm.to_csv('synthetic_Normalized.txt', sep='\t', index=False)

# Standardization
for column in dataStand.columns:
    dataStand[column] = (dataStand[column] - dataStand[column].mean()) / dataStand[column].std()
dataStand.to_csv('turbine_Standardized.txt', sep='\t', index=False)
