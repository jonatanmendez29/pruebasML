import json
import boto3
import pandas as pd
import io
from src.features.build_features import build_features
from src.data.preprocessing import preprocess_data

s3 = boto3.client('s3')


def lambda_handler(event, context):
    """Lambda function for feature engineering"""

    bucket = event['Records'][0]['s3']['bucket']['name']
    key = event['Records'][0]['s3']['object']['key']

    # Only process raw data files
    if not key.startswith('demand-forecast/raw/'):
        return {'status': 'skipped', 'reason': 'Not a raw data file'}

    try:
        # Extract product ID from file name
        product_id = key.split('/')[-1].replace('.parquet', '')

        print(f"Processing features for {product_id}")

        # Download raw data
        response = s3.get_object(Bucket=bucket, Key=key)
        raw_df = pd.read_parquet(io.BytesIO(response['Body'].read()))

        # Preprocess data (using our existing function!)
        processed_df = preprocess_data(raw_df)

        # Build features (using our existing function!)
        features_df = build_features(product_id, df=processed_df)

        # Upload processed features to S3
        output_key = f"demand-forecast/processed/{product_id}.parquet"
        buffer = io.BytesIO()
        features_df.to_parquet(buffer, index=False)
        buffer.seek(0)

        s3.upload_fileobj(buffer, bucket, output_key)

        print(f"Completed feature engineering for {product_id}")

        return {
            'statusCode': 200,
            'body': json.dumps({
                'status': 'success',
                'product_id': product_id,
                'processed_rows': len(features_df)
            })
        }

    except Exception as e:
        print(f"Error processing {key}: {str(e)}")
        return {
            'statusCode': 500,
            'body': json.dumps({
                'status': 'error',
                'error': str(e)
            })
        }