#!/usr/bin/env python3
"""
Preprocessing script for Nextflow workflow.
Takes raw data and creates processed features for model training.
"""

import pandas as pd
import numpy as np
import sys
from sklearn.preprocessing import StandardScaler

def preprocess_data(input_file, output_file):
    print(f"Preprocessing {input_file} -> {output_file}")
    
    # Load data
    data = pd.read_csv(input_file)
    
    # Feature engineering
    data['income_log'] = np.log1p(data['income'])
    data['days_inactive_ratio'] = data['days_since_last_purchase'] / (data['purchase_frequency'] + 1)
    data['engagement_score'] = data['website_visits'] * data['purchase_frequency']
    
    # One-hot encode age groups
    data['age_group'] = pd.cut(data['age'], bins=[0, 25, 35, 50, 65, 100], 
                               labels=['18-25', '26-35', '36-50', '51-65', '65+'])
    age_dummies = pd.get_dummies(data['age_group'], prefix='age_group')
    data = pd.concat([data, age_dummies], axis=1)
    
    # Normalize numerical features
    num_features = ['income_log', 'purchase_frequency', 'website_visits', 
                    'days_since_last_purchase', 'days_inactive_ratio', 'engagement_score']
    
    scaler = StandardScaler()
    data[num_features] = scaler.fit_transform(data[num_features])
    
    # Drop original columns we no longer need
    data = data.drop(['age_group', 'income'], axis=1)
    
    # Save processed data
    data.to_csv(output_file, index=False)
    print(f"Saved processed data with shape {data.shape}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python preprocess.py <input_file> <output_file>")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    preprocess_data(input_file, output_file)