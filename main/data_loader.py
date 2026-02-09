"""
Data Loader Module
Load and explore the Telecom Customer Churn dataset
"""

import pandas as pd
import os

def load_data(data_path: str = None) -> pd.DataFrame:
    """Load the customer churn dataset"""
    if data_path is None:
        # Default path relative to project
        data_path = os.path.join(os.path.dirname(__file__), '..', 'dataset', 'telecom_customer_churn.csv')
    
    df = pd.read_csv(data_path)
    print(f"✓ Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    return df

def explore_data(df: pd.DataFrame) -> dict:
    """Basic EDA on the dataset"""
    info = {
        'shape': df.shape,
        'columns': list(df.columns),
        'dtypes': df.dtypes.value_counts().to_dict(),
        'missing_values': df.isnull().sum().sum(),
        'target_distribution': df['Customer Status'].value_counts().to_dict()
    }
    
    print("\n" + "="*50)
    print("DATASET OVERVIEW")
    print("="*50)
    print(f"Total samples: {info['shape'][0]}")
    print(f"Total features: {info['shape'][1]}")
    print(f"Missing values: {info['missing_values']}")
    print(f"\nTarget Distribution (Customer Status):")
    for status, count in info['target_distribution'].items():
        pct = count / info['shape'][0] * 100
        print(f"  - {status}: {count} ({pct:.1f}%)")
    
    return info

if __name__ == "__main__":
    df = load_data()
    explore_data(df)
