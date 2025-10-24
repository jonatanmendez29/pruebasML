# Part 4: MLOps & Production Readiness
## 1. **CI/CD for ML**:
- "Design a CI/CD pipeline for ML models"
- How would you automate testing of model quality? 
- Discuss your approach to reproducible builds 

```yaml
# .github/workflows/ml-pipeline.yml
name: ML Training Pipeline

on:
  push:
    branches: [main]
  schedule:
    - cron: '0 2 * * *'  # Daily retraining

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - run: pytest tests/ --cov=src/
      - run: python -m pylint src/
  
  train:
    needs: test
    runs-on: ubuntu-latest
    steps:
      - run: python train.py --environment staging
      - run: python evaluate.py --threshold 0.75
  
  deploy:
    needs: train
    runs-on: ubuntu-latest
    if: needs.train.outputs.metrics_improved == 'true'
    steps:
      - run: python deploy.py --environment canary --traffic-percent 10
```
## 2. **Monitoring & Observability**:
- What metrics would you monitor in production? 
- How do you detect data drift and concept drift? 
- Discuss your experience with MLflow, Airflow, or similar tools

```python
# Key metrics to monitor
PRODUCTION_METRICS = {
    # Infrastructure metrics
    'cpu_utilization': 'AWS/SageMaker',
    'memory_usage': 'AWS/SageMaker',
    'invocation_latency': 'AWS/SageMaker',
    
    # Data quality metrics
    'feature_drift': 'Custom/PopulationStabilityIndex',
    'missing_value_rate': 'Custom/DataQuality',
    
    # Model performance metrics
    'prediction_drift': 'Custom/PredictionDistribution',
    'business_metrics': 'Custom/ConversionRate',
    
    # Business impact
    'revenue_impact': 'Custom/BusinessValue',
    'user_engagement': 'Custom/SessionDuration'
}
```
## 3. Shell Scripting & Automation:
- "Write a shell script to automate model retraining pipeline"
- How would you schedule and monitor periodic jobs?

```bash
#!/bin/bash
set -e

# Configuration
MODEL_NAME="recommendation-v1"
S3_BUCKET="my-ml-models"
STAGE=${1:-staging}

echo "Starting model retraining pipeline for $MODEL_NAME"

# Data validation
echo "Validating training data..."
python scripts/validate_data.py \
    --input-s3 "s3://$S3_BUCKET/data/training/" \
    --output-s3 "s3://$S3_BUCKET/data/validated/"

# Feature engineering
echo "Running feature engineering..."
python scripts/feature_engineering.py \
    --input-s3 "s3://$S3_BUCKET/data/validated/" \
    --output-s3 "s3://$S3_BUCKET/features/"

# Model training
echo "Training model..."
python train.py \
    --features-s3 "s3://$S3_BUCKET/features/" \
    --model-s3 "s3://$S3_BUCKET/models/$MODEL_NAME/" \
    --hyperparameters config/hyperparams.json

# Model evaluation
echo "Evaluating model..."
python evaluate.py \
    --model-s3 "s3://$S3_BUCKET/models/$MODEL_NAME/" \
    --test-s3 "s3://$S3_BUCKET/data/test/" \
    --output-s3 "s3://$S3_BUCKET/evaluation/$MODEL_NAME/"

# Check if model meets deployment criteria
EVAL_RESULT=$(python scripts/check_metrics.py \
    --metrics-s3 "s3://$S3_BUCKET/evaluation/$MODEL_NAME/metrics.json")

if [ "$EVAL_RESULT" == "PASS" ]; then
    echo "Model passed evaluation, deploying to $STAGE"
    python deploy.py --model-name $MODEL_NAME --stage $STAGE
else
    echo "Model failed evaluation: $EVAL_RESULT"
    exit 1
fi

echo "Retraining pipeline completed successfully"
```