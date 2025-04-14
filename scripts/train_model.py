#!/usr/bin/env python3
"""
Model training script for Nextflow workflow.
Takes processed data and trains a machine learning model.
"""

import pandas as pd
import numpy as np
import sys
import json
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

def train_model(processed_data_file, random_seed, model_output_file, metrics_output_file):
    print(f"Training model with data from {processed_data_file}")
    print(f"Using random seed: {random_seed}")
    
    # Load processed data
    data = pd.read_csv(processed_data_file)
    
    # Separate features and target
    X = data.drop(['customer_id', 'churn'], axis=1)
    y = data['churn']
    
    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=int(random_seed)
    )
    
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    
    # Train model
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=5,
        random_state=int(random_seed)
    )
    
    model.fit(X_train, y_train)
    
    # Evaluate model
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    metrics = {
        'accuracy': float(accuracy_score(y_test, y_pred)),
        'precision': float(precision_score(y_test, y_pred)),
        'recall': float(recall_score(y_test, y_pred)),
        'f1_score': float(f1_score(y_test, y_pred)),
        'roc_auc': float(roc_auc_score(y_test, y_prob)),
        'feature_importance': dict(zip(X.columns.tolist(), 
                                     model.feature_importances_.tolist()))
    }
    
    # Save model
    with open(model_output_file, 'wb') as f:
        pickle.dump(model, f)
    
    # Save metrics
    with open(metrics_output_file, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print(f"Model saved to {model_output_file}")
    print(f"Metrics saved to {metrics_output_file}")
    print(f"Model performance: accuracy={metrics['accuracy']:.4f}, f1={metrics['f1_score']:.4f}")

if __name__ == "__main__":
    if len(sys.argv) != 5:
        print("Usage: python train_model.py <processed_data_file> <random_seed> <model_output_file> <metrics_output_file>")
        sys.exit(1)
    
    processed_data_file = sys.argv[1]
    random_seed = sys.argv[2]
    model_output_file = sys.argv[3]
    metrics_output_file = sys.argv[4]
    
    train_model(processed_data_file, random_seed, model_output_file, metrics_output_file)