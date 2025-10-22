import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import json
import sys

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from utils.config import load_config
from features.build_features import build_features


def load_model(product_id):
    """Load trained model and feature information"""
    models_dir = Path('models')
    model_path = models_dir / f'model_{product_id}.joblib'
    feature_path = models_dir / f'feature_info_{product_id}.json'

    if not model_path.exists():
        raise FileNotFoundError(f"Model for product {product_id} not found. Please train the model first.")

    model = joblib.load(model_path)

    with open(feature_path, 'r') as f:
        feature_info = json.load(f)

    return model, feature_info


def predict_future(model, feature_info, product_id, periods=30):
    """
    Generate future predictions

    Parameters:
    - model: trained model
    - feature_info: feature information dictionary
    - product_id: product ID
    - periods: number of periods to forecast
    """
    config = load_config()

    # Get the latest data for the product
    df_current = build_features(product_id)
    df_current = df_current.sort_values('date').tail(60)  # Last 60 days

    # Generate predictions
    predictions = []
    current_features = df_current.copy()

    feature_columns = feature_info['feature_columns']

    for i in range(periods):
        # Get the most recent row
        latest_row = current_features.iloc[-1:].copy()

        # Update date
        next_date = latest_row['date'].iloc[0] + pd.Timedelta(days=1)
        latest_row['date'] = next_date

        # Update temporal features
        latest_row['day_of_week'] = next_date.dayofweek
        latest_row['month'] = next_date.month
        latest_row['is_weekend'] = int(next_date.dayofweek >= 5)

        # Update cyclical features
        latest_row['day_of_week_sin'] = np.sin(2 * np.pi * latest_row['day_of_week'] / 7)
        latest_row['day_of_week_cos'] = np.cos(2 * np.pi * latest_row['day_of_week'] / 7)
        latest_row['month_sin'] = np.sin(2 * np.pi * latest_row['month'] / 12)
        latest_row['month_cos'] = np.cos(2 * np.pi * latest_row['month'] / 12)

        # For demonstration, we'll assume no promotion and no holiday
        # In practice, you'd get this from a business calendar
        latest_row['promotion'] = 0
        latest_row['holiday'] = 0

        # Prepare features for prediction
        prediction_features = latest_row[feature_columns]

        # Make prediction
        pred = model.predict(prediction_features)[0]
        predictions.append(pred)

        # Update the latest_row with the prediction for future lag features
        # This is a simplified approach - in practice, you'd update all lag features
        latest_row['units_sold'] = pred

        # Append to current features for next iteration
        current_features = pd.concat([current_features, latest_row], ignore_index=True)

    # Create result DataFrame
    future_dates = pd.date_range(
        start=df_current['date'].max() + pd.Timedelta(days=1),
        periods=periods,
        freq='D'
    )

    result_df = pd.DataFrame({
        'date': future_dates,
        'product_id': product_id,
        'predicted_demand': predictions
    })

    return result_df


def predict_single_date(model, feature_info, product_id, prediction_date, features_dict=None):
    """
    Predict demand for a specific date

    Parameters:
    - model: trained model
    - feature_info: feature information dictionary
    - product_id: product ID
    - prediction_date: date to predict for
    - features_dict: dictionary of additional features (promotion, holiday, etc.)
    """
    if features_dict is None:
        features_dict = {}

    # Get base features from the product's recent data
    df_current = build_features(product_id)
    latest_data = df_current.sort_values('date').iloc[-1:].copy()

    # Update with prediction date and provided features
    latest_data['date'] = prediction_date
    latest_data['day_of_week'] = prediction_date.dayofweek
    latest_data['month'] = prediction_date.month
    latest_data['is_weekend'] = int(prediction_date.dayofweek >= 5)

    # Update cyclical features
    latest_data['day_of_week_sin'] = np.sin(2 * np.pi * latest_data['day_of_week'] / 7)
    latest_data['day_of_week_cos'] = np.cos(2 * np.pi * latest_data['day_of_week'] / 7)
    latest_data['month_sin'] = np.sin(2 * np.pi * latest_data['month'] / 12)
    latest_data['month_cos'] = np.cos(2 * np.pi * latest_data['month'] / 12)

    # Update with provided features
    for key, value in features_dict.items():
        if key in latest_data.columns:
            latest_data[key] = value

    # Prepare features for prediction
    feature_columns = feature_info['feature_columns']
    prediction_features = latest_data[feature_columns]

    # Make prediction
    prediction = model.predict(prediction_features)[0]

    return {
        'product_id': product_id,
        'date': prediction_date,
        'predicted_demand': prediction,
        'features_used': feature_columns
    }


if __name__ == "__main__":
    # Example usage
    model, feature_info = load_model('P003')

    # Predict next 30 days
    future_predictions = predict_future(model, feature_info, 'P003', periods=30)
    print("Future predictions:")
    print(future_predictions.head(10))

    # Predict for a specific date
    specific_date = pd.Timestamp('2024-01-01')
    single_pred = predict_single_date(
        model,
        feature_info,
        'P003',
        specific_date,
        features_dict={'promotion': 1, 'holiday': 0}
    )
    print(f"\nPrediction for {specific_date.date()}: {single_pred['predicted_demand']:.1f} units")