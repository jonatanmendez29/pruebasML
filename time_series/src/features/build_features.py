import numpy as np
import pandas as pd
from pathlib import Path
import sys

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from data.make_dataset import load_config, load_raw_data, save_processed_data
from utils.config import load_config


def create_temporal_features(df):
    """Create temporal features from date column"""
    df = df.copy()

    # Basic date features
    df['day_of_week'] = df['date'].dt.dayofweek
    df['month'] = df['date'].dt.month
    df['quarter'] = df['date'].dt.quarter
    df['year'] = df['date'].dt.year
    df['is_weekend'] = (df['date'].dt.dayofweek >= 5).astype(int)

    # Cyclical encoding
    config = load_config()
    if 'day_of_week' in config['features']['cyclical_features']:
        df['day_of_week_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['day_of_week_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)

    if 'month' in config['features']['cyclical_features']:
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)

        # Holiday proximity (days until next major holiday)

    def days_to_holiday(date):
        holidays = {
            'christmas': pd.Timestamp(f'{date.year}-12-25'),
            'new_year': pd.Timestamp(f'{date.year + 1}-01-01'),
            'july_4': pd.Timestamp(f'{date.year}-07-04')
        }
        min_days = 365
        for holiday in holidays.values():
            days = abs((date - holiday).days)
            min_days = min(min_days, days)
        return min_days

    df['days_to_holiday'] = df['date'].apply(days_to_holiday)

    return df


def build_features(product_id=None):
    """Main function to build features for all products or specific product"""
    config = load_config()

    # Load and create basic features
    df = load_raw_data()
    df = create_temporal_features(df)

    # If specific product, create lag features
    if product_id:
        df_product = df[df['product_id'] == product_id].copy()
        df_product = create_lag_features(df_product, config)
        save_processed_data(df_product, f'features_{product_id}.parquet')
        return df_product
    else:
        # For all products (simplified - in practice, you'd process each product)
        save_processed_data(df, 'features_all.parquet')
        return df


def create_lag_features(df, config=None):
    """
    Create lag and rolling window features for time series data

    Parameters:
    - df: pandas DataFrame for a single product
    - config: configuration dictionary (optional)
    """
    if config is None:
        config = load_config()

    df_lagged = df.copy().sort_values('date')

    # Get parameters from config
    lag_periods = config['features']['lag_periods']
    window_sizes = config['features']['window_sizes']
    rolling_stats = config['features']['rolling_features']

    target_col = config['training']['target_column']

    # Create lag features
    for lag in lag_periods:
        df_lagged[f'lag_{lag}'] = df_lagged[target_col].shift(lag)

    # Create rolling window statistics
    for window in window_sizes:
        for stat in rolling_stats:
            if stat == 'mean':
                df_lagged[f'rolling_{stat}_{window}'] = (
                    df_lagged[target_col].shift(1).rolling(window=window, min_periods=1).mean()
                )
            elif stat == 'std':
                df_lagged[f'rolling_{stat}_{window}'] = (
                    df_lagged[target_col].shift(1).rolling(window=window, min_periods=1).std()
                )
            elif stat == 'max':
                df_lagged[f'rolling_{stat}_{window}'] = (
                    df_lagged[target_col].shift(1).rolling(window=window, min_periods=1).max()
                )
            elif stat == 'min':
                df_lagged[f'rolling_{stat}_{window}'] = (
                    df_lagged[target_col].shift(1).rolling(window=window, min_periods=1).min()
                )

    # Create price change features
    df_lagged['price_change_1d'] = df_lagged['selling_price'].pct_change(1)
    df_lagged['price_change_7d'] = df_lagged['selling_price'].pct_change(7)

    # Create momentum-like features
    for window in [7, 14]:
        df_lagged[f'momentum_{window}'] = (
                                                  df_lagged[target_col].shift(1) /
                                                  df_lagged[target_col].shift(window)
                                          ) - 1

    # Create seasonality features (same period last year)
    df_lagged['same_period_last_year'] = df_lagged[target_col].shift(365)

    return df_lagged


def create_cross_category_features(df_all, target_product_id):
    """
    Create features based on category-level performance
    """
    # Get category of target product
    target_category = df_all[df_all['product_id'] == target_product_id]['category'].iloc[0]

    # Calculate category-level aggregates
    category_agg = df_all[df_all['category'] == target_category].groupby('date').agg({
        'units_sold': ['mean', 'sum']
    }).reset_index()

    category_agg.columns = ['date', 'category_mean_demand', 'category_total_demand']

    # Merge with original data
    df_with_category = pd.merge(
        df_all[df_all['product_id'] == target_product_id],
        category_agg,
        on='date',
        how='left'
    )

    # Create relative performance features
    df_with_category['demand_vs_category'] = (
            df_with_category['units_sold'] / df_with_category['category_mean_demand']
    )

    return df_with_category

if __name__ == "__main__":
    # Build features for our star product
    df_features = build_features('P003')
    print(df_features)
    print(f"Built features with shape: {df_features.shape}")