#!/usr/bin/env python3
"""
Script to generate example dataset for Nextflow workflow testing.
Creates a synthetic dataset of customer information with features and a target variable.
"""

import numpy as np
import pandas as pd
import os

# Create data directories if they don't exist
os.makedirs("data", exist_ok=True)

# Set random seed for reproducibility
np.random.seed(42)

# Generate synthetic data
n_samples = 1000

# Features
age = np.random.normal(45, 15, n_samples).astype(int)
age = np.clip(age, 18, 90)  # Clip to reasonable age range

income = np.random.lognormal(10.5, 0.5, n_samples).astype(int)
purchase_frequency = np.random.poisson(3, n_samples)
website_visits = np.random.negative_binomial(5, 0.5, n_samples)
days_since_last_purchase = np.random.exponential(30, n_samples).astype(int)

# Create a latent variable for target correlation
latent = 0.02 * age - 0.000001 * income + 0.2 * purchase_frequency - 0.05 * days_since_last_purchase + np.random.normal(0, 1, n_samples)

# Target: customer churn (0 or 1)
churn_prob = 1 / (1 + np.exp(-latent))  # Sigmoid to get probability
churn = (np.random.random(n_samples) < churn_prob).astype(int)

# Create DataFrame
data = pd.DataFrame({
    'customer_id': [f'CUST_{i:05d}' for i in range(n_samples)],
    'age': age,
    'income': income,
    'purchase_frequency': purchase_frequency,
    'website_visits': website_visits,
    'days_since_last_purchase': days_since_last_purchase,
    'churn': churn
})

# Save to CSV
data.to_csv("data/raw_data.csv", index=False)

print(f"Created dataset with {n_samples} samples at data/raw_data.csv")
print("Data preview:")
print(data.head())
print("\nData statistics:")
print(data.describe())