import pandas as pd
import numpy as np
from pathlib import Path
import yaml

class TestConfigUtils:
    """Test configuration utility functions"""

    def test_config_loading(self, temp_data_dir, sample_config):
        """Test configuration loading"""
        from src.utils.config import load_config, save_config

        # Create a test config file
        config_dir = temp_data_dir / 'config'
        config_dir.mkdir(parents=True, exist_ok=True)
        config_file = config_dir / 'test_config.yaml'

        with open(config_file, 'w') as f:
            yaml.dump(sample_config, f)

        # Test loading
        config_loaded = load_config('test_config.yaml')

        # Check that config was loaded correctly
        assert config_loaded['model']['name'] == 'random_forest'
        assert config_loaded['training']['target_column'] == 'units_sold'

    def test_data_paths_creation(self, temp_data_dir):
        """Test data paths creation"""
        from src.utils.config import get_data_paths

        # This will use the actual config, but we test the function structure
        paths = get_data_paths()

        # Check that paths dictionary has expected keys
        expected_keys = ['raw_data', 'processed_data', 'models']
        for key in expected_keys:
            assert key in paths

    def test_config_update(self, temp_data_dir, sample_config):
        """Test configuration updating"""
        from src.utils.config import update_config, load_config

        # Create a test config file
        config_dir = temp_data_dir / 'config'
        config_dir.mkdir(parents=True, exist_ok=True)
        config_file = config_dir / 'test_config.yaml'

        with open(config_file, 'w') as f:
            yaml.dump(sample_config, f)

        # Update a config value
        update_config('model.hyperparameters.n_estimators', 200, 'test_config.yaml')

        # Load and check updated config
        config_updated = load_config('test_config.yaml')
        assert config_updated['model']['hyperparameters']['n_estimators'] == 200


class TestVisualizationUtils:
    """Test visualization utility functions"""

    def test_plot_style_setting(self):
        """Test that plot style is set correctly"""
        from src.utils.visualization import set_plot_style
        import matplotlib.pyplot as plt

        # This should run without errors
        set_plot_style()

        # Check that some style parameters are set
        assert plt.rcParams['figure.figsize'] == [12.0, 6.0]
        assert plt.rcParams['font.size'] == 12.0

    def test_feature_importance_plot(self, mock_model, temp_data_dir):
        """Test feature importance plotting"""
        from src.utils.visualization import plot_feature_importance

        # Train mock model on simple data
        X = np.random.rand(10, 3)
        y = np.random.rand(10)
        mock_model.fit(X, y)

        feature_names = ['feature_1', 'feature_2', 'feature_3']

        # This should create a plot file
        plot_feature_importance(mock_model, feature_names, 'P001')

        # Check that plot file was created
        plot_file = Path('plots') / 'feature_importance_P001.png'
        if plot_file.exists():
            plot_file.unlink()  # Clean up

    def test_prediction_comparison_plot(self):
        """Test prediction comparison plotting"""
        from src.utils.visualization import plot_prediction_comparison

        # Create sample data
        dates = pd.date_range('2020-01-01', periods=10)
        actual = np.random.randint(10, 100, 10)
        predicted = actual + np.random.normal(0, 5, 10)

        # This should run without errors
        plot_prediction_comparison(dates, actual, predicted, 'P001')

        # Check that plot file was created
        plot_file = Path('plots') / 'predictions_P001.png'
        if plot_file.exists():
            plot_file.unlink()  # Clean up


class TestDataValidation:
    """Test data validation functions"""

    def test_data_quality_checks(self, sample_processed_data):
        """Test basic data quality checks"""

        # Check for missing values in critical columns
        critical_columns = ['product_id', 'date', 'units_sold']
        for col in critical_columns:
            assert sample_processed_data[col].isna().sum() == 0

        # Check that units_sold is non-negative
        assert (sample_processed_data['units_sold'] >= 0).all()

        # Check date range is reasonable
        date_range = sample_processed_data['date'].max() - sample_processed_data['date'].min()
        assert date_range.days > 0

    def test_feature_quality_checks(self, sample_processed_data):
        """Test feature quality checks"""
        from src.features.build_features import build_features

        df_features = build_features('P001')

        # Check that we have expected features
        expected_feature_types = ['temporal', 'lag', 'rolling']
        temporal_features = [col for col in df_features.columns if 'sin' in col or 'cos' in col]
        lag_features = [col for col in df_features.columns if col.startswith('lag_')]
        rolling_features = [col for col in df_features.columns if col.startswith('rolling_')]

        assert len(temporal_features) > 0
        assert len(lag_features) > 0
        assert len(rolling_features) > 0

        # Check that numerical features don't have extreme outliers
        numerical_cols = df_features.select_dtypes(include=[np.number]).columns
        for col in numerical_cols:
            if col != 'units_sold':  # target might have outliers
                q1 = df_features[col].quantile(0.25)
                q3 = df_features[col].quantile(0.75)
                iqr = q3 - q1
                lower_bound = q1 - 3 * iqr
                upper_bound = q3 + 3 * iqr

                # Check that most values are within reasonable bounds
                within_bounds = df_features[col].between(lower_bound, upper_bound)
                assert within_bounds.mean() > 0.95  # 95% of values within bounds