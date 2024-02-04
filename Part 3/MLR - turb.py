import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import numpy as np

# Load data from the text file
df = pd.read_csv('turbine_Standardized.txt', sep='\t')

# Split the data into features and target
X = df.drop('power_of_hydroelectrical_turbine', axis=1)
y = df['power_of_hydroelectrical_turbine']

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a Linear Regression model
model = LinearRegression()

# Train the model
model.fit(X_train, y_train)

# Predict the test set results
y_pred = model.predict(X_test)

# Print the model score
print('Model score:', model.score(X_test, y_test))

# Calculate the Mean Absolute Percentage Error (MAPE)
mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
print('Mean Absolute Percentage Error (MAPE): ', mape)

# Create a scatter plot of the real vs predicted values
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.3)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
plt.xlabel('Real Value')
plt.ylabel('Predicted Value')
plt.title('Real vs Predicted Values for Linear Regression')
plt.show()
