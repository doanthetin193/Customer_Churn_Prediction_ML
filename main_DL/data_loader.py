"""
Data loader for the DL pipeline.
"""

import os
import pandas as pd


def load_data(data_path: str | None = None) -> pd.DataFrame:
    """Load telecom churn dataset from CSV."""
    if data_path is None:
        data_path = os.path.join(
            os.path.dirname(__file__), "..", "dataset", "telecom_customer_churn.csv"
        )

    df = pd.read_csv(data_path)
    print(f"Loaded dataset: {df.shape[0]} rows, {df.shape[1]} columns")
    return df


def explore_data(df: pd.DataFrame) -> dict:
    """Print and return quick dataset summary."""
    info = {
        "shape": df.shape,
        "columns": list(df.columns),
        "dtypes": df.dtypes.value_counts().to_dict(),
        "missing_values": int(df.isnull().sum().sum()),
        "target_distribution": df["Customer Status"].value_counts().to_dict(),
    }

    print("\n" + "=" * 50)
    print("DATASET OVERVIEW")
    print("=" * 50)
    print(f"Total samples: {info['shape'][0]}")
    print(f"Total features: {info['shape'][1]}")
    print(f"Missing values: {info['missing_values']}")
    print("\nTarget distribution (Customer Status):")

    for status, count in info["target_distribution"].items():
        pct = count / info["shape"][0] * 100
        print(f"  - {status}: {count} ({pct:.1f}%)")

    return info


if __name__ == "__main__":
    frame = load_data()
    explore_data(frame)
