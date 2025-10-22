import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
import joblib
from pathlib import Path
import json
from datetime import datetime
import sys

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from data.make_dataset import load_config
from features.build_features import build_features
from utils.visualization import plot_feature_importance, plot_prediction_comparison


def prepare_training_data(df, target_col='units_sold'):
    """Prepare features and target for training"""
    config = load_config()

    # Define feature columns (dynamically based on available columns)
    base_features = [
        'day_of_week_sin', 'day_of_week_cos', 'month_sin', 'month_cos',
        'is_weekend', 'promotion', 'holiday', 'days_to_holiday'
    ]

    # Add lag features that exist in the dataframe
    lag_features = [col for col in df.columns if col.startswith('lag_') or col.startswith('rolling_')]
    price_features = [col for col in df.columns if 'price_change' in col]
    additional_features = [col for col in df.columns if col in ['momentum_7', 'momentum_14', 'same_period_last_year']]

    feature_columns = base_features + lag_features + price_features + additional_features

    # Remove rows with NaN values
    df_clean = df.dropna(subset=feature_columns + [target_col])

    # Split chronologically
    split_date = config['training']['validation_split']
    train_mask = df_clean['date'] < split_date
    test_mask = df_clean['date'] >= split_date

    X_train = df_clean[train_mask][feature_columns]
    X_test = df_clean[test_mask][feature_columns]
    y_train = df_clean[train_mask][target_col]
    y_test = df_clean[test_mask][target_col]

    return X_train, X_test, y_train, y_test, feature_columns


def cross_validate_timeseries(model, X, y, n_splits=5):
    """Perform time series cross-validation"""
    tscv = TimeSeriesSplit(n_splits=n_splits)
    cv_scores = []

    for train_idx, val_idx in tscv.split(X):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)

        mae = mean_absolute_error(y_val, y_pred)
        cv_scores.append(mae)

    return np.mean(cv_scores), np.std(cv_scores)


def train_model(product_id='P001', save_model=True):
    """Main training function"""
    config = load_config()

    print(f"Training model for product {product_id}...")

    # Build features
    df_features = build_features(product_id)

    # Prepare training data
    X_train, X_test, y_train, y_test, feature_columns = prepare_training_data(df_features)

    print(f"Training samples: {len(X_train)}, Test samples: {len(X_test)}")
    print(f"Number of features: {len(feature_columns)}")

    # Initialize and train model
    model_config = config['model']['hyperparameters']

    if config['model']['name'] == 'random_forest':
        model = RandomForestRegressor(**model_config)
    else:
        raise ValueError(f"Unsupported model: {config['model']['name']}")

    # Perform cross-validation
    print("Performing time series cross-validation...")
    cv_mean, cv_std = cross_validate_timeseries(model, X_train, y_train)
    print(f"Cross-validation MAE: {cv_mean:.2f} ± {cv_std:.2f}")

    # Train final model on all training data
    model.fit(X_train, y_train)

    # Generate predictions and calculate metrics
    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100

    metrics = {
        'mae': float(mae),
        'mse': float(mse),
        'rmse': float(rmse),
        'mape': float(mape),
        'cv_mae_mean': float(cv_mean),
        'cv_mae_std': float(cv_std)
    }

    print(f"\nModel Performance for {product_id}:")
    for metric, value in metrics.items():
        print(f"{metric.upper()}: {value:.2f}")

    if save_model:
        # Save model
        models_dir = Path('models')
        models_dir.mkdir(exist_ok=True)

        model_path = models_dir / f'model_{product_id}.joblib'
        joblib.dump(model, model_path)

        # Save feature names
        feature_info = {
            'feature_columns': feature_columns,
            'feature_importance': dict(zip(feature_columns, model.feature_importances_)),
            'training_date': datetime.now().isoformat(),
            'metrics': metrics
        }

        feature_path = models_dir / f'feature_info_{product_id}.json'
        with open(feature_path, 'w') as f:
            json.dump(feature_info, f, indent=2)

        print(f"Model saved to {model_path}")
        print(f"Feature info saved to {feature_path}")

    # Create visualizations
    plot_feature_importance(model, feature_columns, product_id)

    test_dates = df_features[df_features['date'] >= config['training']['validation_split']]['date']
    plot_prediction_comparison(test_dates, y_test, y_pred, product_id)

    return model, feature_columns, metrics


if __name__ == "__main__":
    model, features, metrics = train_model('P003')