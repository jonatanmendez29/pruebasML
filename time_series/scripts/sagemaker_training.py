import argparse
import json
import pandas as pd
from src.features.build_features import build_features
from src.models.train import prepare_training_data, train_model
import os
import joblib


def train_sagemaker():
    """Training script adapted for SageMaker environment"""

    # SageMaker passes paths via environment variables
    input_path = os.environ.get('SM_CHANNEL_TRAINING', '/opt/ml/input/data/training')
    model_path = os.environ.get('SM_MODEL_DIR', '/opt/ml/model')

    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--product-id', type=str, required=True)
    parser.add_argument('--config-path', type=str, default='/opt/ml/input/config')
    args = parser.parse_args()

    print(f"Training model for product: {args.product_id}")
    print(f"Input path: {input_path}, Model path: {model_path}")

    try:
        # Load data (in real scenario, this comes from S3 via SageMaker channel)
        data_file = f"{input_path}/{args.product_id}.parquet"
        df = pd.read_parquet(data_file)

        # Build features and train model (using our existing functions!)
        df_features = build_features(args.product_id, df=df)
        model, feature_columns, metrics = train_model(
            args.product_id,
            save_model=False,
            df_features=df_features
        )

        # Save model in SageMaker format
        model_file = f"{model_path}/model.joblib"
        joblib.dump(model, model_file)

        # Save feature information
        metadata = {
            'product_id': args.product_id,
            'feature_columns': feature_columns,
            'metrics': metrics,
            'feature_importance': dict(zip(feature_columns, model.feature_importances_))
        }

        with open(f"{model_path}/metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)

        print(f"Training completed for {args.product_id}")
        print(f"Metrics: {metrics}")

    except Exception as e:
        print(f"Error training model for {args.product_id}: {str(e)}")
        raise


if __name__ == "__main__":
    train_sagemaker()