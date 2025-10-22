import pandas as pd
import numpy as np

class TestDataProcessing:
    """Test data loading and preprocessing functionality"""

    def test_load_raw_data(self, temp_data_dir, sample_raw_data):
        """Test loading raw data"""
        from src.data.make_dataset import load_raw_data, save_processed_data

        # Save sample data to temp directory
        test_file = temp_data_dir / 'test_data.csv'
        sample_raw_data.to_csv(test_file, index=False)

        # Test loading
        df_loaded = load_raw_data()
        assert isinstance(df_loaded, pd.DataFrame)
        assert len(df_loaded) > 0

    def test_preprocessing_handles_missing_values(self, sample_raw_data):
        """Test that preprocessing handles missing values correctly"""
        from src.data.preprocessing import preprocess_data, handle_missing_values

        # Introduce some missing values
        df_with_nan = sample_raw_data.copy()
        df_with_nan.loc[10:15, 'units_sold'] = np.nan

        # Test handling missing values
        df_clean = handle_missing_values(df_with_nan, method='forward_fill')
        assert df_clean['units_sold'].isna().sum() == 0

    def test_preprocessing_removes_outliers(self, sample_raw_data):
        """Test outlier removal functionality"""
        from src.data.preprocessing import remove_outliers_iqr

        # Add an obvious outlier
        df_with_outlier = sample_raw_data.copy()
        outlier_index = df_with_outlier.index[10]
        df_with_outlier.loc[outlier_index, 'units_sold'] = 1000

        df_clean = remove_outliers_iqr(df_with_outlier, 'units_sold')

        # Check that outlier was removed
        assert df_clean['units_sold'].max() < 1000
        assert len(df_clean) < len(df_with_outlier)

    def test_preprocessing_pipeline(self, sample_raw_data):
        """Test the complete preprocessing pipeline"""
        from src.data.preprocessing import preprocess_data

        df_processed = preprocess_data(sample_raw_data)

        # Check basic properties
        assert isinstance(df_processed, pd.DataFrame)
        assert len(df_processed) > 0
        assert 'product_id' in df_processed.columns
        assert 'units_sold' in df_processed.columns
        assert df_processed['units_sold'].isna().sum() == 0

    def test_data_types_after_preprocessing(self, sample_raw_data):
        """Test that data types are correct after preprocessing"""
        from src.data.preprocessing import preprocess_data

        df_processed = preprocess_data(sample_raw_data)

        # Check data types
        assert pd.api.types.is_categorical_dtype(df_processed['product_id'])
        assert pd.api.types.is_categorical_dtype(df_processed['category'])
        assert pd.api.types.is_datetime64_any_dtype(df_processed['date'])

    def test_data_integrity(self, sample_raw_data):
        """Test that preprocessing maintains data integrity"""
        from src.data.preprocessing import preprocess_data

        original_shape = sample_raw_data.shape
        df_processed = preprocess_data(sample_raw_data)

        # Should not lose all data
        assert len(df_processed) > 0
        # Should have same or fewer rows (due to outlier removal)
        assert len(df_processed) <= original_shape[0]
        # Should have same number of key columns
        assert 'product_id' in df_processed.columns
        assert 'units_sold' in df_processed.columns