import boto3
import json
import pandas as pd
from typing import Dict, List, Any


class ForecastingEndpoint:
    """Manage real-time inference endpoints"""

    def __init__(self, endpoint_name: str, s3_bucket: str):
        self.sagemaker_runtime = boto3.client('sagemaker-runtime')
        self.sagemaker = boto3.client('sagemaker')
        self.endpoint_name = endpoint_name
        self.s3_bucket = s3_bucket

    def create_endpoint(self, model_name: str, instance_type: str = 'ml.m5.large'):
        """Create a real-time inference endpoint"""

        endpoint_config_name = f"{model_name}-config"

        # Create endpoint configuration
        self.sagemaker.create_endpoint_config(
            EndpointConfigName=endpoint_config_name,
            ProductionVariants=[
                {
                    'VariantName': 'primary',
                    'ModelName': model_name,
                    'InitialInstanceCount': 1,
                    'InstanceType': instance_type,
                    'InitialVariantWeight': 1.0
                }
            ]
        )

        # Create endpoint
        try:
            self.sagemaker.create_endpoint(
                EndpointName=self.endpoint_name,
                EndpointConfigName=endpoint_config_name
            )
            print(f"✅ Endpoint {self.endpoint_name} creation initiated")
        except self.sagemaker.exceptions.ClientError as e:
            if "AlreadyExists" in str(e):
                print(f"✅ Endpoint {self.endpoint_name} already exists")
            else:
                raise

    def predict_single(self, product_id: str, features: Dict[str, Any]) -> float:
        """Make real-time prediction for a single product"""

        # Prepare payload
        payload = {
            'product_id': product_id,
            'features': features
        }

        try:
            response = self.sagemaker_runtime.invoke_endpoint(
                EndpointName=self.endpoint_name,
                ContentType='application/json',
                Body=json.dumps(payload)
            )

            result = json.loads(response['Body'].read().decode())
            return result['prediction']

        except Exception as e:
            print(f"❌ Prediction failed: {e}")
            # Fallback to historical average
            return self._fallback_prediction(product_id)

    def batch_predict(self, product_ids: List[str], prediction_date: str) -> pd.DataFrame:
        """Generate batch predictions for multiple products"""

        # This would typically use SageMaker Batch Transform
        # For simplicity, we'll simulate the process
        predictions = []

        for product_id in product_ids:
            # In reality, we'd get features from feature store
            features = self._get_features_for_prediction(product_id, prediction_date)

            prediction = self.predict_single(product_id, features)

            predictions.append({
                'product_id': product_id,
                'prediction_date': prediction_date,
                'predicted_demand': prediction,
                'confidence_interval': self._calculate_confidence(prediction)
            })

        return pd.DataFrame(predictions)

    def _fallback_prediction(self, product_id: str) -> float:
        """Fallback logic when model prediction fails"""
        # Simple historical average as fallback
        from s3_data_manager import S3DataManager
        s3_mgr = S3DataManager(self.s3_bucket)

        try:
            historical_data = s3_mgr.download_dataframe(f"processed/{product_id}.parquet")
            return historical_data['units_sold'].mean()
        except:
            return 50.0  # Default fallback

    def _get_features_for_prediction(self, product_id: str, date: str) -> Dict[str, Any]:
        """Get features for prediction (simplified)"""
        # In production, this would query a feature store
        return {
            'day_of_week': pd.Timestamp(date).dayofweek,
            'month': pd.Timestamp(date).month,
            'is_weekend': int(pd.Timestamp(date).dayofweek >= 5),
            'promotion': 0,  # Would come from business calendar
            'holiday': 0,  # Would come from holiday calendar
            'lag_1': 45.0,  # Would come from recent data
            'rolling_mean_7': 42.0
        }


# Real-time inference handler
def lambda_inference_handler(event, context):
    """Lambda function for real-time inference"""

    endpoint = ForecastingEndpoint('demand-forecast-endpoint', 'your-bucket')

    try:
        # Parse request
        product_id = event['product_id']
        features = event.get('features', {})

        # Make prediction
        prediction = endpoint.predict_single(product_id, features)

        return {
            'statusCode': 200,
            'body': json.dumps({
                'product_id': product_id,
                'predicted_demand': prediction,
                'timestamp': pd.Timestamp.now().isoformat(),
                'model_version': '1.0'
            })
        }

    except Exception as e:
        return {
            'statusCode': 500,
            'body': json.dumps({
                'error': str(e),
                'fallback_used': True
            })
        }
