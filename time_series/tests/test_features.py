import pandas as pd

class TestFeatureEngineering:
    """Test feature engineering functionality"""

    def test_temporal_features_creation(self, sample_product_features):
        """Test creation of temporal features"""
        from src.features.build_features import create_temporal_features

        df_with_features = create_temporal_features(sample_product_features)

        # Check that temporal features are created
        expected_features = ['day_of_week', 'month', 'is_weekend']
        for feature in expected_features:
            assert feature in df_with_features.columns

        # Check cyclical encoding
        assert 'day_of_week_sin' in df_with_features.columns
        assert 'day_of_week_cos' in df_with_features.columns
        assert 'month_sin' in df_with_features.columns
        assert 'month_cos' in df_with_features.columns

    def test_lag_features_creation(self, sample_product_features):
        """Test creation of lag features"""
        from src.features.lag_features import create_lag_features

        # Add temporal features first
        from src.features.build_features import create_temporal_features
        df_with_temp = create_temporal_features(sample_product_features)

        # Create lag features
        df_with_lags = create_lag_features(df_with_temp)

        # Check lag features
        expected_lags = ['lag_1', 'lag_7']
        for lag in expected_lags:
            assert lag in df_with_lags.columns

        # Check rolling features
        assert 'rolling_mean_7' in df_with_lags.columns
        assert 'rolling_std_7' in df_with_lags.columns

        # Check that first rows have NaN for lag features (as expected)
        assert df_with_lags['lag_1'].isna().sum() > 0

    def test_price_change_features(self, sample_product_features):
        """Test creation of price change features"""
        from src.features.lag_features import create_lag_features

        # Add temporal features first
        from src.features.build_features import create_temporal_features
        df_with_temp = create_temporal_features(sample_product_features)

        # Create lag features (which includes price change features)
        df_with_features = create_lag_features(df_with_temp)

        # Check price change features
        assert 'price_change_1d' in df_with_features.columns
        assert 'price_change_7d' in df_with_features.columns

    def test_feature_building_pipeline(self, sample_processed_data):
        """Test the complete feature building pipeline"""
        from src.features.build_features import build_features

        # Test building features for a specific product
        df_features = build_features('P001')

        # Check that features are created
        assert isinstance(df_features, pd.DataFrame)
        assert len(df_features) > 0

        # Check that we have both temporal and lag features
        assert 'day_of_week_sin' in df_features.columns
        assert 'lag_1' in df_features.columns
        assert 'rolling_mean_7' in df_features.columns

    def test_feature_dtypes(self, sample_processed_data):
        """Test that feature data types are correct"""
        from src.features.build_features import build_features

        df_features = build_features('P001')

        # Check that numerical features are numerical
        numerical_features = ['lag_1', 'lag_7', 'rolling_mean_7', 'day_of_week_sin']
        for feature in numerical_features:
            if feature in df_features.columns:
                assert pd.api.types.is_numeric_dtype(df_features[feature])

    def test_no_data_leakage_in_features(self, sample_processed_data):
        """Test that feature engineering doesn't cause data leakage"""
        from src.features.build_features import build_features

        df_features = build_features('P001')

        # Check that lag features don't use future information
        # This is ensured by using shift() in the implementation
        # We can verify by checking that the first row has NaN for lag features
        first_row = df_features.iloc[0]
        lag_columns = [col for col in df_features.columns if col.startswith('lag_')]

        for lag_col in lag_columns:
            if lag_col in first_row:
                # First row should have NaN for lag features
                assert pd.isna(first_row[lag_col])