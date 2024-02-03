import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
import numpy as np

# Define your configurations
configs = [
    {'layers': (10,), 'lr': 0.01, 'momentum': 0.9, 'activation': 'relu', 'epochs': 1000},
    {'layers': (20, 10), 'lr': 0.01, 'momentum': 0.9, 'activation': 'relu', 'epochs': 1000},
    {'layers': (30, 20, 10), 'lr': 0.01, 'momentum': 0.9, 'activation': 'relu', 'epochs': 1000},
]

# Load data from the text file
df = pd.read_csv('synthetic_Normalized.txt', sep='\t')

# Split the data into features and target
X = df.drop('z', axis=1)
y = df['z']

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Iterate over configurations
for config in configs:
    # Create a Multilayer Perceptron regressor
    model = MLPRegressor(hidden_layer_sizes=config['layers'], learning_rate_init=config['lr'],
                         momentum=config['momentum'], activation=config['activation'],
                         max_iter=config['epochs'], random_state=42)

    # Train the regressor
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
    plt.title(f'Real vs Predicted Values for {config}')
    plt.show()
