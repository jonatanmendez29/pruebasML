import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import tempfile

# Add the project root to Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.data.make_dataset import generate_retail_demand_data

@pytest.fixture
def temp_data_dir():
    """Create a temporary directory for test data"""
    with tempfile.TemporaryDirectory() as tmp_dir:
        yield Path(tmp_dir)

@pytest.fixture
def sample_raw_data():
    """Generate sample raw data for testing"""
    return generate_retail_demand_data(
        start_date='2022-01-01',
        end_date='2024-12-31',
        random_seed=42
    )

@pytest.fixture
def sample_processed_data(sample_raw_data):
    """Generate sample processed data for testing"""
    from src.data.preprocessing import preprocess_data
    return preprocess_data(sample_raw_data)

@pytest.fixture
def sample_product_features():
    """Create sample feature data for a single product"""
    dates = pd.date_range('2022-01-01', '2022-12-31', freq='D')
    data = {
        'date': dates,
        'product_id': 'P001',
        'units_sold': np.random.randint(10, 100, len(dates)),
        'promotion': np.random.choice([0, 1], len(dates), p=[0.9, 0.1]),
        'holiday': np.random.choice([0, 1], len(dates), p=[0.95, 0.05]),
        'selling_price': 100.0,
    }
    df = pd.DataFrame(data)
    return df

@pytest.fixture
def sample_config():
    """Return sample configuration for testing"""
    return {
        'data': {
            'raw_data_path': 'data/raw/test_data.csv',
            'processed_path': 'data/processed/',
            'test_size': 0.2
        },
        'features': {
            'lag_periods': [1, 7],
            'window_sizes': [7],
            'rolling_features': ['mean', 'std'],
            'cyclical_features': ['day_of_week', 'month']
        },
        'model': {
            'name': 'random_forest',
            'hyperparameters': {
                'n_estimators': 10,
                'max_depth': 5,
                'random_state': 42
            }
        },
        'training': {
            'target_column': 'units_sold',
            'validation_split': '2024-01-01',
            'metrics': ['mae', 'mse']
        }
    }

@pytest.fixture
def mock_model():
    """Create a mock model for testing"""
    from sklearn.ensemble import RandomForestRegressor
    return RandomForestRegressor(n_estimators=5, random_state=42)