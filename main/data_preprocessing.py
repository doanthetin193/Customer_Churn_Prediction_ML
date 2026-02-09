"""
Data Preprocessing Module
Clean, encode, and prepare data for ML models
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

def create_target_variable(df: pd.DataFrame) -> pd.DataFrame:
    """Create binary target: 1 = Churned, 0 = Not Churned"""
    df = df.copy()
    df['Churn'] = (df['Customer Status'] == 'Churned').astype(int)
    return df

def select_features(df: pd.DataFrame) -> pd.DataFrame:
    """Select relevant features for modeling (matching paper)"""
    # Features to use (based on paper)
    feature_cols = [
        'Gender', 'Age', 'Married', 'Number of Dependents',
        'Number of Referrals', 'Tenure in Months', 'Offer',
        'Phone Service', 'Multiple Lines', 'Internet Service', 'Internet Type',
        'Online Security', 'Online Backup', 'Device Protection Plan',
        'Premium Tech Support', 'Streaming TV', 'Streaming Movies',
        'Streaming Music', 'Unlimited Data', 'Contract', 'Paperless Billing',
        'Payment Method', 'Monthly Charge', 'Total Charges', 'Total Revenue'
    ]
    
    available_cols = [c for c in feature_cols if c in df.columns]
    return df[available_cols + ['Churn']]

def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Handle missing values"""
    df = df.copy()
    
    # Fill numeric columns with median
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if df[col].isnull().sum() > 0:
            df[col].fillna(df[col].median(), inplace=True)
    
    # Fill categorical columns with mode
    cat_cols = df.select_dtypes(include=['object']).columns
    for col in cat_cols:
        if df[col].isnull().sum() > 0:
            df[col].fillna(df[col].mode()[0], inplace=True)
    
    print(f"✓ Missing values handled")
    return df

def encode_categorical(df: pd.DataFrame) -> tuple:
    """Encode categorical variables using LabelEncoder"""
    df = df.copy()
    encoders = {}
    
    cat_cols = df.select_dtypes(include=['object']).columns
    
    for col in cat_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        encoders[col] = le
    
    print(f"✓ Encoded {len(cat_cols)} categorical columns")
    return df, encoders

def scale_features(X_train: pd.DataFrame, X_test: pd.DataFrame) -> tuple:
    """Scale numerical features using StandardScaler"""
    scaler = StandardScaler()
    
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print(f"✓ Features scaled")
    return X_train_scaled, X_test_scaled, scaler

def preprocess_data(df: pd.DataFrame, test_size: float = 0.25, random_state: int = 42, use_smote: bool = False) -> dict:
    """
    Complete preprocessing pipeline
    Returns dict with X_train, X_test, y_train, y_test and metadata
    """
    print("\n" + "="*50)
    print("DATA PREPROCESSING")
    print("="*50)
    
    # Step 1: Create target variable
    df = create_target_variable(df)
    
    # Step 2: Select features
    df = select_features(df)
    
    # Step 3: Handle missing values
    df = handle_missing_values(df)
    
    # Step 4: Encode categorical variables
    df, encoders = encode_categorical(df)
    
    # Step 5: Split features and target
    X = df.drop('Churn', axis=1)
    y = df['Churn']
    
    feature_names = list(X.columns)
    
    # Step 6: Train/Test split (75/25 as per paper)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    print(f"✓ Data split: Train={len(X_train)}, Test={len(X_test)}")
    
    # Step 7: Scale features
    X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)
    
    # Step 8: Handle imbalanced data with SMOTE (optional)
    if use_smote:
        try:
            from imblearn.over_sampling import SMOTE
            smote = SMOTE(random_state=random_state)
            X_train_scaled, y_train_resampled = smote.fit_resample(X_train_scaled, y_train)
            y_train = y_train_resampled
            print(f"✓ SMOTE applied: Train size increased to {len(X_train_scaled)}")
        except ImportError:
            print("⚠ SMOTE not available (install imbalanced-learn). Using original data.")
    
    print(f"\n✓ Preprocessing complete!")
    print(f"  Train set: {X_train_scaled.shape}")
    print(f"  Test set: {X_test_scaled.shape}")
    if hasattr(y_train, 'mean'):
        print(f"  Churn rate (train): {y_train.mean()*100:.1f}%")
    else:
        print(f"  Churn rate (train): {np.mean(y_train)*100:.1f}%")
    
    return {
        'X_train': X_train_scaled,
        'X_test': X_test_scaled,
        'y_train': y_train if isinstance(y_train, np.ndarray) else y_train.values,
        'y_test': y_test.values,
        'feature_names': feature_names,
        'encoders': encoders,
        'scaler': scaler,
        'X_train_df': X_train,  # Keep original for explainability
        'X_test_df': X_test
    }

if __name__ == "__main__":
    from data_loader import load_data
    df = load_data()
    data = preprocess_data(df)
