# Import necessary libraries
import pandas as pd

# Load the datasets
dataN = pd.read_csv('A1-synthetic.txt', sep='\t')
dataS = pd.read_csv('A1-turbine.txt', sep='\t')

# Normalization
dataNorm = (dataN-dataN.min())/(dataN.max()-dataN.min())
dataNorm.to_csv('synthetic_Normalized.txt', sep='\t', index=False)

# Standardization
dataStand = (dataS-dataS.mean())/dataS.std()
dataStand.to_csv('turbine_Standardized.txt', sep='\t', index=False)
