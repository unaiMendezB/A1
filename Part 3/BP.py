import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

# Load data from the text file
data = pd.read_csv('synthetic_Normalized.txt', sep='\t')

# Split the data into features and target
X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

# Standardize the features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a Multilayer Perceptron regressor
reg = MLPRegressor(hidden_layer_sizes=(100,), max_iter=300, alpha=1e-4,
                   solver='sgd', verbose=10, random_state=42,
                   learning_rate_init=.1)

# Train the regressor
reg.fit(X_train, y_train)

# Predict the test set results
y_pred = reg.predict(X_test)

# Calculate the Mean Absolute Percentage Error (MAPE)
mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
print('Mean Absolute Percentage Error (MAPE): ', mape)

# Create a scatter plot of the real vs predicted values
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.3)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
plt.xlabel('Real Value')
plt.ylabel('Predicted Value')
plt.title('Real vs Predicted Values')
plt.show()
