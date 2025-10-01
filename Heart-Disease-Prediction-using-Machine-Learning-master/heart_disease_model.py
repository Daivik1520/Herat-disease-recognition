#!/usr/bin/env python3
# Heart Disease Prediction Model
# Created by Daivik Reddy
# A simplified version of the Jupyter notebook implementation

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load the dataset
print("Loading heart disease dataset...")
dataset = pd.read_csv("heart.csv")
print(f"Dataset shape: {dataset.shape}")

# Display the first few rows
print("\nFirst 5 rows of the dataset:")
print(dataset.head())

# Basic information about the dataset
print("\nDataset information:")
print(dataset.describe().T)

# Check for missing values
print("\nMissing values in dataset:")
print(dataset.isnull().sum())

# Prepare the data
X = dataset.drop('target', axis=1)
y = dataset['target']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"\nTraining set size: {X_train.shape}")
print(f"Testing set size: {X_test.shape}")

# Feature scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train the Random Forest model (best performer from the notebook)
print("\nTraining Random Forest model...")
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Make predictions
y_pred = rf_model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"\nModel Accuracy: {accuracy:.4f}")

# Display classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Display confusion matrix
print("\nConfusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
print(cm)

# Feature importance
feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': rf_model.feature_importances_
}).sort_values('Importance', ascending=False)

print("\nFeature Importance:")
print(feature_importance)

# Make a prediction for a sample patient
print("\nPrediction for a sample patient:")
# Sample data: 52-year-old male with chest pain type 2, blood pressure 125, etc.
sample = np.array([[52, 1, 2, 125, 212, 0, 1, 168, 0, 1.0, 2, 2, 3]])
sample_scaled = scaler.transform(sample)
prediction = rf_model.predict(sample_scaled)
prediction_proba = rf_model.predict_proba(sample_scaled)

print(f"Prediction (0=No Disease, 1=Disease): {prediction[0]}")
print(f"Probability of heart disease: {prediction_proba[0][1]:.4f}")

print("\nHeart Disease Prediction Model completed successfully!")