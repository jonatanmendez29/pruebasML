import boto3
from src.cloud.s3_data_manager import S3DataManager
from stepfunction_orchestrator import TrainingOrchestrator

def deploy_cloud_infrastructure():
    """Deploy complete cloud forecasting infrastructure"""

    # Configuration
    config = {
        'bucket_name': 'your-demand-forecast-bucket',
        'region': 'us-east-1',
        'role_arn': 'your-sagemaker-execution-role'
    }

    print("🚀 Deploying Cloud Forecasting Infrastructure...")

    # 1. Create S3 bucket
    s3 = boto3.client('s3', region_name=config['region'])
    try:
        s3.create_bucket(
            Bucket=config['bucket_name'],
            CreateBucketConfiguration={'LocationConstraint': config['region']}
        )
        print("✅ S3 bucket created")
    except s3.exceptions.BucketAlreadyExists:
        print("✅ S3 bucket already exists")

    # 2. Upload initial data
    s3_manager = S3DataManager(config['bucket_name'])
    from src.data.make_dataset import generate_retail_demand_data
    df = generate_retail_demand_data()

    # Upload sample products
    sample_products = ['P001', 'P002', 'P003']
    for product_id in sample_products:
        product_data = df[df['product_id'] == product_id]
        s3_manager.upload_dataframe(product_data, f"raw/{product_id}.parquet")

    print("✅ Sample data uploaded")

    # 3. Create Step Functions orchestration
    orchestrator = TrainingOrchestrator(config['bucket_name'], config['role_arn'])
    state_machine_arn = orchestrator.create_training_state_machine()
    print("✅ Step Functions state machine created")

    # 4. Test the pipeline
    print("🧪 Testing pipeline with sample products...")
    execution_arn = orchestrator.start_training_pipeline(sample_products)
    print(f"✅ Pipeline started: {execution_arn}")

    return {
        'bucket_name': config['bucket_name'],
        'state_machine_arn': state_machine_arn,
        'execution_arn': execution_arn
    }


if __name__ == "__main__":
    results = deploy_cloud_infrastructure()
    print("\n🎉 Deployment Complete!")
    print(f"S3 Bucket: {results['bucket_name']}")
    print(f"State Machine: {results['state_machine_arn']}")
    print(f"Execution: {results['execution_arn']}")