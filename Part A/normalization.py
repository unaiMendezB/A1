# Import necessary libraries
import pandas as pd

# Load the datasets
dataSynth = pd.read_csv('A1-synthetic.txt', sep='\t')
dataTurb = pd.read_csv('A1-turbine.txt', sep='\t')

# Normalization
for column in dataSynth.columns:
    dataSynth[column] = (dataSynth[column] - dataSynth[column].min()) / (dataSynth[column].max() - dataSynth[column].min())

# Standardization
for column in dataTurb.columns:
    dataTurb[column] = (dataTurb[column] - dataTurb[column].mean()) / dataTurb[column].std()

# Split index
indexSynth = int(dataSynth.shape[0] * 0.8)
indexTurb = int(dataTurb.shape[0] * 0.85)

# Split the data
trainSynth = dataSynth.iloc[:indexSynth]
testSynth = dataSynth.iloc[indexSynth:]

trainTurb = dataTurb.iloc[:indexTurb]
testTurb = dataTurb.iloc[indexTurb:]

# Save the split data into two new files
trainSynth.to_csv('synthetic_Normalized.txt', sep='\t', index=False)
testSynth.to_csv('synthetic_Normalized_TEST.txt', sep='\t', index=False)
trainTurb.to_csv('turbine_Standardized.txt', sep='\t', index=False)
testTurb.to_csv('turbine_Standardized_TEST.txt', sep='\t', index=False)