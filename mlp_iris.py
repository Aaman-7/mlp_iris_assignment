#!/usr/bin/env python3

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report

import pandas as pd

# 1. Load Iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# Display information
print("Iris dataset loaded successfully.")
print(f"Number of samples: {X.shape[0]}")
print(f"Number of features: {X.shape[1]}")
print(f"Target classes: {iris.target_names}")

# 2. Preprocess the data
# Encode target variable
encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)

# Split the dataset into training and testing setss
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.3, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("\nData preprocessing complete.")
print(f"Training set size: {X_train_scaled.shape[0]}")
print(f"Test set size: {X_test_scaled.shape[0]}")

# 3. Initialize and Train the MLPClassifier
# Define the MLP model
# hidden_layer_sizes: tuple, i-th element represents the number of neurons in the i-th hidden layer.
# max_iter: Maximum number of iterations for the solver to converge.
# alpha: L2 penalty parameter.
# solver: The solver for weight optimization. adam
# random_state: For reproducibility.
mlp = MLPClassifier(hidden_layer_sizes=(100,), max_iter=500, alpha=1e-4, solver='adam', random_state=42)

print("\nTraining the MLPClassifier...")
mlp.fit(X_train_scaled, y_train)
print("MLPClassifier training complete.")

# 4. Make predictions and evaluate the model
print("\nMaking predictions on the test set...")
y_pred = mlp.predict(X_test_scaled)

print("\nModel Evaluation:")
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=iris.target_names))

# 5. Discuss potential improvements or further analysis.
print("\nDiscussion points for further analysis or improvements:")
print("- Hyperparameter tuning (e.g., hidden_layer_sizes, activation, solver, alpha, learning_rate)")
print("- Cross-validation to get a more robust estimate of model performance.")
print("- Investigating misclassified samples.")
print("- Feature importance analysis (though less straightforward for MLPs).")
print("- Early stopping to prevent overfitting.")