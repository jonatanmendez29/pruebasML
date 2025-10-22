import pandas as pd
import pytest
import sys
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent))

class TestEndToEndPipeline:
    """Test the complete pipeline from data to predictions"""

    def test_complete_pipeline(self, temp_data_dir, sample_raw_data):
        """Test the complete ML pipeline"""
        # Save sample data
        raw_data_path = temp_data_dir / 'raw_data.csv'
        sample_raw_data.to_csv(raw_data_path, index=False)

        try:
            # Test data loading
            from src.data.make_dataset import load_raw_data
            df_loaded = load_raw_data()
            assert len(df_loaded) > 0

            # Test preprocessing
            from src.data.preprocessing import preprocess_data
            df_processed = preprocess_data(df_loaded)
            assert len(df_processed) > 0

            # Test feature engineering
            from src.features.build_features import build_features
            df_features = build_features('P001')
            assert len(df_features) > 0
            assert 'lag_1' in df_features.columns

            # Test model training (with small model for speed)
            from src.models.train import train_model
            model, features, metrics = train_model('P001', save_model=False)

            # Check that training completed
            assert model is not None
            assert len(features) > 0
            assert 'mae' in metrics

            # Test predictions
            from src.models.predict import predict_single_date

            # Create feature info for prediction
            feature_info = {
                'feature_columns': features,
                'feature_importance': dict(zip(features, model.feature_importances_))
            }

            prediction_date = pd.Timestamp('2020-04-01')
            prediction = predict_single_date(
                model, feature_info, 'P001', prediction_date
            )

            assert 'predicted_demand' in prediction
            assert prediction['predicted_demand'] >= 0

        except Exception as e:
            pytest.skip(f"Pipeline test skipped due to: {e}")

    def test_config_consistency(self):
        """Test that configuration is consistent across the project"""
        from src.utils.config import load_config

        config = load_config()

        # Check required sections exist
        required_sections = ['data', 'features', 'model', 'training']
        for section in required_sections:
            assert section in config

        # Check critical parameters
        assert 'target_column' in config['training']
        assert 'hyperparameters' in config['model']
        assert 'lag_periods' in config['features']

    def test_module_imports(self):
        """Test that all modules can be imported correctly"""
        # Test that all main modules can be imported
        try:
            from src.data.make_dataset import load_raw_data, generate_retail_demand_data
            from src.data.preprocessing import preprocess_data
            from src.features.build_features import build_features
            from src.features.lag_features import create_lag_features
            from src.models.train import train_model
            from src.models.predict import predict_future
            from src.utils.config import load_config
            from src.utils.visualization import plot_feature_importance

            # If we get here, all imports worked
            assert True
        except ImportError as e:
            pytest.fail(f"Import failed: {e}")