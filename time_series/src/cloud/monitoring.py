import pandas as pd
import numpy as np
import boto3
import json
from datetime import datetime
from typing import Dict, List, Any

from s3_data_manager import S3DataManager


class ModelMonitor:
    """Monitor model performance and data drift"""

    def __init__(self, s3_bucket: str):
        self.s3 = boto3.client('s3')
        self.cloudwatch = boto3.client('cloudwatch')
        self.s3_bucket = s3_bucket

    def calculate_model_metrics(self, y_true: List[float], y_pred: List[float]) -> Dict[str, float]:
        """Calculate model performance metrics"""

        mae = np.mean(np.abs(np.array(y_true) - np.array(y_pred)))
        mape = np.mean(np.abs((np.array(y_true) - np.array(y_pred)) / np.array(y_true))) * 100
        rmse = np.sqrt(np.mean((np.array(y_true) - np.array(y_pred)) ** 2))

        return {
            'mae': float(mae),
            'mape': float(mape),
            'rmse': float(rmse),
            'sample_size': len(y_true)
        }

    def detect_data_drift(self, current_data: pd.DataFrame, reference_data: pd.DataFrame) -> Dict[str, Any]:
        """Detect data drift between current and reference data"""

        drift_metrics = {}

        numerical_cols = current_data.select_dtypes(include=[np.number]).columns

        for col in numerical_cols:
            if col in ['units_sold']:  # Skip target variable
                continue

            # KS test for distribution change
            from scipy import stats
            statistic, p_value = stats.ks_2samp(
                reference_data[col].dropna(),
                current_data[col].dropna()
            )

            drift_metrics[col] = {
                'ks_statistic': statistic,
                'p_value': p_value,
                'drift_detected': p_value < 0.05  # Significant drift
            }

        return drift_metrics

    def check_concept_drift(self, product_id: str, lookback_days: int = 30) -> Dict[str, Any]:
        """Check for concept drift by monitoring recent performance"""

        try:
            # Load recent predictions and actuals
            recent_data = self._load_recent_predictions(product_id, lookback_days)

            if len(recent_data) < 10:  # Not enough data
                return {'status': 'insufficient_data'}

            metrics = self.calculate_model_metrics(
                recent_data['actual_demand'],
                recent_data['predicted_demand']
            )

            # Check if performance has degraded
            performance_degraded = metrics['mape'] > 15.0  # Threshold

            return {
                'status': 'degraded' if performance_degraded else 'healthy',
                'metrics': metrics,
                'threshold_violated': performance_degraded
            }

        except Exception as e:
            return {'status': 'error', 'error': str(e)}

    def publish_metrics_to_cloudwatch(self, metrics: Dict[str, float], product_id: str):
        """Publish metrics to CloudWatch for alerting"""

        metric_data = []
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        for metric_name, value in metrics.items():
            metric_data.append({
                'MetricName': metric_name,
                'Dimensions': [
                    {'Name': 'ProductId', 'Value': product_id},
                    {'Name': 'ModelType', 'Value': 'DemandForecast'}
                ],
                'Value': value,
                'Unit': 'Count',
                'Timestamp': timestamp
            })

        try:
            self.cloudwatch.put_metric_data(
                Namespace='DemandForecast',
                MetricData=metric_data
            )
        except Exception as e:
            print(f"Failed to publish metrics: {e}")

    def create_alerts(self, product_id: str):
        """Create CloudWatch alerts for model monitoring"""

        alarm_name = f"DemandForecast-Drift-{product_id}"

        self.cloudwatch.put_metric_alarm(
            AlarmName=alarm_name,
            AlarmDescription=f'Data drift detected for {product_id}',
            MetricName='mape',
            Namespace='DemandForecast',
            Statistic='Average',
            Dimensions=[
                {'Name': 'ProductId', 'Value': product_id},
                {'Name': 'ModelType', 'Value': 'DemandForecast'}
            ],
            Period=300,  # 5 minutes
            EvaluationPeriods=2,
            Threshold=15.0,  # MAPE threshold
            ComparisonOperator='GreaterThanThreshold',
            AlarmActions=[
                'arn:aws:sns:us-east-1:123456789012:model-drift-alerts'  # Your SNS topic
            ]
        )

    def _load_recent_predictions(self, product_id: str, lookback_days: int = 30) -> pd.DataFrame:
        """Load recent predictions and actuals for drift detection"""

        try:
            from s3_data_manager import S3DataManager
            s3_mgr = S3DataManager(self.s3_bucket)

            # Try to load monitoring data for this product
            monitoring_key = f"monitoring/{product_id}_predictions.parquet"

            try:
                predictions_df = s3_mgr.download_dataframe(monitoring_key)

                # Filter for recent data
                recent_cutoff = pd.Timestamp.now() - pd.Timedelta(days=lookback_days)
                recent_data = predictions_df[
                    predictions_df['timestamp'] >= recent_cutoff
                    ].copy()

                return recent_data

            except Exception as e:
                print(f"No existing monitoring data for {product_id}, generating sample data")
                # Generate sample monitoring data for demonstration
                return self._generate_sample_monitoring_data(product_id, lookback_days)

        except Exception as e:
            print(f"Error loading recent predictions for {product_id}: {e}")
            # Return empty DataFrame with expected columns
            return pd.DataFrame(columns=['timestamp', 'predicted_demand', 'actual_demand'])

    def _generate_sample_monitoring_data(self, product_id: str, lookback_days: int) -> pd.DataFrame:
        """Generate sample monitoring data for demonstration purposes"""

        dates = pd.date_range(
            start=pd.Timestamp.now() - pd.Timedelta(days=lookback_days),
            end=pd.Timestamp.now(),
            freq='D'
        )

        # Create realistic sample data with some noise and potential drift
        base_demand = 50
        predictions = []
        actuals = []

        # Simulate some concept drift in the second half of the period
        for i, date in enumerate(dates):
            # Base prediction with some error
            prediction = base_demand + np.random.normal(0, 5)

            # Actual demand with potential drift
            actual = base_demand + np.random.normal(0, 5)

            # Introduce gradual concept drift in second half
            if i > len(dates) // 2:
                # Simulate changing patterns (e.g., seasonality change)
                actual += 10 * np.sin(2 * np.pi * i / 7)  # Weekly pattern emerges
                prediction_error = np.random.normal(5, 2)  # Systematic bias develops
                actual += prediction_error

            predictions.append(max(1, prediction))
            actuals.append(max(1, actual))

        monitoring_data = pd.DataFrame({
            'timestamp': dates,
            'product_id': product_id,
            'predicted_demand': predictions,
            'actual_demand': actuals,
            'absolute_error': np.abs(np.array(predictions) - np.array(actuals)),
            'percentage_error': (np.abs(np.array(predictions) - np.array(actuals)) / np.array(actuals)) * 100
        })

        # Save the generated sample data for future use
        try:
            from s3_data_manager import S3DataManager
            s3_mgr = S3DataManager(self.s3_bucket)
            s3_mgr.upload_dataframe(
                monitoring_data,
                f"monitoring/{product_id}_predictions.parquet"
            )
            print(f"✅ Generated and saved sample monitoring data for {product_id}")
        except Exception as e:
            print(f"Note: Could not save sample data: {e}")

        return monitoring_data

    def record_prediction_result(self, product_id: str, prediction: float,
                                 actual: float, timestamp: str = None):
        """Record prediction results for monitoring and drift detection"""

        if timestamp is None:
            timestamp = pd.Timestamp.now().isoformat()

        try:
            from s3_data_manager import S3DataManager
            s3_mgr = S3DataManager(self.s3_bucket)

            # Load existing monitoring data or create new
            monitoring_key = f"monitoring/{product_id}_predictions.parquet"

            try:
                existing_data = s3_mgr.download_dataframe(monitoring_key)
            except:
                existing_data = pd.DataFrame(columns=[
                    'timestamp', 'product_id', 'predicted_demand',
                    'actual_demand', 'absolute_error', 'percentage_error'
                ])

            # Add new record
            new_record = pd.DataFrame([{
                'timestamp': timestamp,
                'product_id': product_id,
                'predicted_demand': prediction,
                'actual_demand': actual,
                'absolute_error': abs(prediction - actual),
                'percentage_error': (abs(prediction - actual) / actual) * 100 if actual > 0 else 0
            }])

            updated_data = pd.concat([existing_data, new_record], ignore_index=True)

            # Remove duplicates and keep only recent data (e.g., last 90 days)
            updated_data = updated_data.drop_duplicates(subset=['timestamp'], keep='last')
            cutoff_date = pd.Timestamp.now() - pd.Timedelta(days=90)
            updated_data = updated_data[updated_data['timestamp'] >= cutoff_date]

            # Save updated monitoring data
            s3_mgr.upload_dataframe(updated_data, monitoring_key)

            print(f"✅ Recorded prediction result for {product_id}")

        except Exception as e:
            print(f"❌ Failed to record prediction result for {product_id}: {e}")
    
    def record_prediction_after_actuals(product_id: str, prediction_date: str):
        """Record predictions once actuals are available (typically called daily)"""

        monitor = ModelMonitor('your-bucket')

        try:
            # In production, you'd get the actual sales from your data warehouse
            # For demonstration, we'll use a simple simulation
            from s3_data_manager import S3DataManager
            s3_mgr = S3DataManager('your-bucket')

            # Get the prediction that was made
            predictions_df = s3_mgr.download_dataframe(f"predictions/{prediction_date}.parquet")
            product_prediction = predictions_df[
                predictions_df['product_id'] == product_id
                ]['predicted_demand'].iloc[0]

            # Get the actual sales (in reality, from your sales database)
            # For demo, we'll simulate actual sales with some noise
            actual_sales = product_prediction + np.random.normal(0, product_prediction * 0.1)
            actual_sales = max(1, actual_sales)  # Ensure positive

            # Record the prediction result
            monitor.record_prediction_result(
                product_id=product_id,
                prediction=product_prediction,
                actual=actual_sales,
                timestamp=prediction_date
            )

        except Exception as e:
            print(f"Error recording prediction result for {product_id}: {e}")


def _send_alert_report(alerts: List[Dict[str, Any]]):
    """Send alert report via SNS or other notification service"""

    sns = boto3.client('sns')

    # Create alert message
    alert_count = len(alerts)
    critical_alerts = [a for a in alerts if a.get('metrics', {}).get('mape', 0) > 20]

    message = f"""
            🚨 DEMAND FORECASTING ALERT REPORT 🚨

            Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}
            Total Alerts: {alert_count}
            Critical Alerts: {len(critical_alerts)}

            ALERT SUMMARY:
            {'-' * 50}

        """

    # Add details for each alert
    for i, alert in enumerate(alerts[:10], 1):  # Limit to first 10 alerts
        product_id = alert['product_id']
        metrics = alert.get('metrics', {})
        mape = metrics.get('mape', 0)

        message += f"{i}. {product_id} - MAPE: {mape:.1f}%\n"

    if len(alerts) > 10:
        message += f"\n... and {len(alerts) - 10} more alerts\n"

    message += f"""
            RECOMMENDED ACTIONS:
            1. Review model performance for above products
            2. Check for data quality issues
            3. Consider retraining affected models
            4. Verify feature pipeline is functioning correctly

            View detailed reports in CloudWatch: https://console.aws.amazon.com/cloudwatch/
        """

    try:
        # Publish to SNS topic
        response = sns.publish(
            TopicArn='arn:aws:sns:us-east-1:123456789012:model-drift-alerts',
            Message=message,
            Subject=f'Demand Forecasting Alert - {alert_count} Issues Detected'
        )
        print(f"✅ Alert report sent via SNS: {response['MessageId']}")

    except Exception as e:
        print(f"❌ Failed to send alert report: {e}")

        # Fallback: log to CloudWatch for manual review
        print("ALERT REPORT (Fallback):")
        print(message)


# Scheduled monitoring function
def scheduled_monitoring_handler(event, context):
    """Lambda function for scheduled model monitoring"""

    monitor = ModelMonitor('your-bucket')
    s3_mgr = S3DataManager('your-bucket')

    # Get all active products
    products = s3_mgr.list_products()

    alerts = []

    for product_id in products[:100]:  # Monitor first 100 products
        try:
            # Check for concept drift
            drift_status = monitor.check_concept_drift(product_id)

            if drift_status.get('status') == 'degraded':
                alerts.append({
                    'product_id': product_id,
                    'issue': 'concept_drift',
                    'metrics': drift_status.get('metrics', {}),
                    'timestamp': datetime.now().isoformat()
                })

                # Publish metrics for alerting
                monitor.publish_metrics_to_cloudwatch(
                    drift_status['metrics'],
                    product_id
                )

        except Exception as e:
            print(f"Monitoring failed for {product_id}: {e}")

    # Send summary report
    if alerts:
        _send_alert_report(alerts)

    return {
        'statusCode': 200,
        'body': json.dumps({
            'products_monitored': len(products),
            'alerts_triggered': len(alerts)
        })
    }
