import numpy as np
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))
from utils.config import load_config


def handle_missing_values(df, method='forward_fill'):
    """
    Handle missing values in the dataset

    Parameters:
    - df: pandas DataFrame
    - method: str, method for handling missing values ('forward_fill', 'interpolate', 'drop')
    """
    df_clean = df.copy()

    if method == 'forward_fill':
        df_clean = df_clean.ffill()
    elif method == 'interpolate':
        df_clean = df_clean.interpolate()
    elif method == 'drop':
        df_clean = df_clean.dropna()
    else:
        raise ValueError(f"Unknown method: {method}")

    # Fill any remaining NaN values with 0 for numerical columns
    numerical_cols = df_clean.select_dtypes(include=[np.number]).columns
    df_clean[numerical_cols] = df_clean[numerical_cols].fillna(0)

    return df_clean


def remove_outliers_iqr(df, column, multiplier=1.5):
    """
    Remove outliers using IQR method

    Parameters:
    - df: pandas DataFrame
    - column: str, column name to check for outliers
    - multiplier: float, IQR multiplier (default 1.5)
    """
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - multiplier * IQR
    upper_bound = Q3 + multiplier * IQR

    df_clean = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

    print(f"Removed {len(df) - len(df_clean)} outliers from {column}")
    return df_clean


def preprocess_data(df, target_column='units_sold'):
    """
    Main preprocessing pipeline

    Parameters:
    - df: pandas DataFrame
    - target_column: str, target variable column name
    """
    config = load_config()

    print("Starting data preprocessing...")
    print(f"Initial shape: {df.shape}")

    # Sort by date and product
    df_processed = df.sort_values(['product_id', 'date']).reset_index(drop=True)

    # Handle missing values
    df_processed = handle_missing_values(df_processed, method='forward_fill')

    # Remove extreme outliers from target variable
    df_processed = remove_outliers_iqr(df_processed, target_column, multiplier=3)

    # Ensure data types are correct
    df_processed['product_id'] = df_processed['product_id'].astype('category')
    df_processed['category'] = df_processed['category'].astype('category')

    print(f"Final shape after preprocessing: {df_processed.shape}")

    return df_processed


if __name__ == "__main__":
    from make_dataset import load_raw_data

    df = load_raw_data()
    df_processed = preprocess_data(df)
    print("Data preprocessing completed successfully!")