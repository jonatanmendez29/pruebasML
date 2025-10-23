import boto3
import pandas as pd
import io
from pathlib import Path


class S3DataManager:
    """Manage data storage and retrieval from S3"""

    def __init__(self, bucket_name: str, prefix: str = "demand-forecast"):
        self.s3 = boto3.client('s3')
        self.bucket = bucket_name
        self.prefix = prefix

    def upload_dataframe(self, df: pd.DataFrame, s3_key: str):
        """Upload DataFrame to S3 as Parquet"""
        buffer = io.BytesIO()
        df.to_parquet(buffer, index=False)
        buffer.seek(0)

        full_key = f"{self.prefix}/{s3_key}"
        self.s3.upload_fileobj(buffer, self.bucket, full_key)
        print(f"Uploaded {len(df)} rows to s3://{self.bucket}/{full_key}")

    def download_dataframe(self, s3_key: str) -> pd.DataFrame:
        """Download DataFrame from S3 Parquet file"""
        full_key = f"{self.prefix}/{s3_key}"
        response = self.s3.get_object(Bucket=self.bucket, Key=full_key)
        return pd.read_parquet(io.BytesIO(response['Body'].read()))

    def list_products(self) -> list:
        """List all available products in S3"""
        response = self.s3.list_objects_v2(
            Bucket=self.bucket,
            Prefix=f"{self.prefix}/raw/"
        )

        products = []
        for obj in response.get('Contents', []):
            if obj['Key'].endswith('.parquet'):
                product_id = Path(obj['Key']).stem
                products.append(product_id)

        return products


# Updated data loading that works with S3
def load_data_cloud(product_id: str, s3_manager: S3DataManager) -> pd.DataFrame:
    """Load data from S3 instead of local files"""
    try:
        s3_key = f"processed/{product_id}.parquet"
        return s3_manager.download_dataframe(s3_key)
    except Exception as e:
        print(f"Error loading {product_id}: {e}")
        # Fallback: generate sample data
        from src.data.make_dataset import generate_retail_demand_data
        df = generate_retail_demand_data()
        product_data = df[df['product_id'] == product_id]
        s3_manager.upload_dataframe(product_data, f"raw/{product_id}.parquet")
        return product_data