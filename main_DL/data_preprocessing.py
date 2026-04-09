"""
Data preprocessing for DL churn prediction.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler


def create_target_variable(df: pd.DataFrame) -> pd.DataFrame:
    """Create binary target where 1 means churned.
    Excludes 'Joined' customers — new customers who have not had a chance to churn yet,
    matching the paper's effective dataset of ~4601 active customers.
    """
    data = df.copy()
    data = data[data["Customer Status"].isin(["Churned", "Stayed"])].reset_index(drop=True)
    data["Churn"] = (data["Customer Status"] == "Churned").astype(int)
    print(f"Filtered to Churned+Stayed only: {len(data)} customers")
    return data


def select_features(df: pd.DataFrame) -> pd.DataFrame:
    """Select features used in the current project for fair comparison."""
    feature_cols = [
        "Gender",
        "Age",
        "Married",
        "Number of Dependents",
        "Number of Referrals",
        "Tenure in Months",
        "Offer",
        "Phone Service",
        "Multiple Lines",
        "Internet Service",
        "Internet Type",
        "Online Security",
        "Online Backup",
        "Device Protection Plan",
        "Premium Tech Support",
        "Streaming TV",
        "Streaming Movies",
        "Streaming Music",
        "Unlimited Data",
        "Contract",
        "Paperless Billing",
        "Payment Method",
        "Monthly Charge",
        "Total Charges",
        "Total Revenue",
    ]

    available_cols = [col for col in feature_cols if col in df.columns]
    return df[available_cols + ["Churn"]]


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Fill numeric NaNs with median and categorical NaNs with mode."""
    data = df.copy()

    numeric_cols = data.select_dtypes(include=[np.number]).columns
    categorical_cols = data.select_dtypes(include=["object"]).columns

    for col in numeric_cols:
        if data[col].isnull().any():
            data[col] = data[col].fillna(data[col].median())

    for col in categorical_cols:
        if data[col].isnull().any():
            data[col] = data[col].fillna(data[col].mode().iloc[0])

    print("Missing values handled")
    return data


def encode_categorical(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, LabelEncoder]]:
    """Label-encode all categorical columns."""
    data = df.copy()
    encoders: dict[str, LabelEncoder] = {}

    categorical_cols = data.select_dtypes(include=["object"]).columns
    for col in categorical_cols:
        encoder = LabelEncoder()
        data[col] = encoder.fit_transform(data[col].astype(str))
        encoders[col] = encoder

    print(f"Encoded {len(categorical_cols)} categorical columns")
    return data, encoders


def scale_features(X_train: pd.DataFrame, X_test: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, StandardScaler]:
    """Scale features using StandardScaler."""
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    print("Features scaled")
    return X_train_scaled, X_test_scaled, scaler


def add_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add domain-informed interaction features for DL improvement.

    Based on SHAP analysis, the most influential features are:
    Contract, Tenure, Monthly Charge, Number of Referrals, Internet Service.
    These interactions let the model learn non-linear relationships directly.
    """
    data = df.copy()
    # Monthly cost burden: high charge on short tenure = churn risk
    data["Monthly_per_Tenure"] = data["Monthly Charge"] / (data["Tenure in Months"] + 1)
    # Average monthly revenue per customer
    data["Revenue_per_Month"] = data["Total Revenue"] / (data["Tenure in Months"] + 1)
    # Referral × Tenure: loyal customers who also refer others
    data["Referral_x_Tenure"] = data["Number of Referrals"] * data["Tenure in Months"]
    # Monthly vs Total charge ratio: detects billing anomalies
    data["Charge_Ratio"] = data["Monthly Charge"] / (data["Total Charges"] + 1)
    return data


def preprocess_data(
    df: pd.DataFrame,
    test_size: float = 0.25,
    random_state: int = 42,
    use_smote: bool = False,
) -> dict:
    """Run full preprocessing pipeline for DL training."""
    print("\n" + "=" * 50)
    print("DATA PREPROCESSING")
    print("=" * 50)

    data = create_target_variable(df)
    data = select_features(data)
    data = handle_missing_values(data)
    data = add_interaction_features(data)
    data, encoders = encode_categorical(data)

    X = data.drop(columns=["Churn"])
    y = data["Churn"]

    feature_names = list(X.columns)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    print(f"Data split: train={len(X_train)}, test={len(X_test)}")

    X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)

    if use_smote:
        try:
            from imblearn.over_sampling import SMOTE

            smote = SMOTE(random_state=random_state)
            X_train_scaled, y_train = smote.fit_resample(X_train_scaled, y_train)
            print(f"SMOTE applied: train size -> {len(X_train_scaled)}")
        except ImportError:
            print("SMOTE not available. Install imbalanced-learn to enable it.")

    y_train_array = y_train if isinstance(y_train, np.ndarray) else y_train.to_numpy()

    print("Preprocessing completed")
    print(f"Train shape: {X_train_scaled.shape}")
    print(f"Test shape: {X_test_scaled.shape}")
    print(f"Train churn rate: {np.mean(y_train_array) * 100:.1f}%")

    return {
        "X_train": X_train_scaled,
        "X_test": X_test_scaled,
        "y_train": y_train_array,
        "y_test": y_test.to_numpy(),
        "feature_names": feature_names,
        "encoders": encoders,
        "scaler": scaler,
        "X_train_df": X_train,
        "X_test_df": X_test,
    }


if __name__ == "__main__":
    from data_loader import load_data

    frame = load_data()
    preprocess_data(frame)
