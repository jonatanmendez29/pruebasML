import pandas as pd
from utils.config import load_config


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