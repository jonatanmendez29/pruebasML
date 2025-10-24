# Part 3: AWS & Distributed Systems**
## 1. Walk me through deploying a model on AWS from development to production
Deployment pipeline:

```shell
#!/bin/bash
# Example deployment script
set -e

# Build and push Docker image
docker build -t $ECR_URL:$GIT_SHA .
docker push $ECR_URL:$GIT_SHA

# Run integration tests
python -m pytest tests/integration/ -v

# Update SageMaker endpoint
aws sagemaker update-endpoint \
    --endpoint-name "recommendations-prod" \
    --endpoint-config-name "recommendations-$(date +%Y%m%d-%H%M%S)"

# Run smoke tests
./scripts/smoke-test.sh

# Update feature store
python scripts/update_feature_store.py --environment prod
```

## 2. How would you design a distributed training system for large datasets?

- **Data Parallelism:** Split data across multiple GPUs using SageMaker Distributed Training
- **Model Parallelism:** For large models that don't fit in single GPU memory
- **Parameter Server:** For synchronous updates with large parameter spaces
- **All-Reduce:** For synchronous training with Horovod or PyTorch DDP

## 3. Compare SageMaker, ECS, and EKS for ML workloads
```
SageMaker vs ECS vs EKS for ML:

SageMaker:
✅ Managed service, less operational overhead
✅ Built-in hyperparameter tuning
✅ Built-in A/B testing capabilities
✅ Optimized ML containers
❌ Less flexibility, vendor lock-in

ECS:
✅ More control over container environment
✅ Better cost control with spot instances
✅ Easier to customize networking
❌ More operational overhead

EKS:
✅ Kubernetes ecosystem and tools
✅ Multi-cloud portability
✅ Advanced scheduling capabilities
❌ Highest operational complexity

Choice: SageMaker for rapid iteration, ECS for cost-sensitive production, EKS for complex multi-model serving
```
## 4. How do you handle model versioning and rollbacks in production?
- **Versioning:** MLflow Model Registry with semantic versioning
- **Artifact Storage:** S3 with versioning enabled for model binaries
- **Rollback Strategy:** Blue-green deployment with SageMaker endpoints
- **Database:** DynamoDB for storing model metadata and performance metrics

## Scenario: Your model inference latency increased by 300% overnight. Walk me through your debugging process.
```
1. Immediate Actions:
   - Check CloudWatch metrics for CPU/Memory utilization
   - Review SageMaker endpoint logs for errors
   - Verify Auto Scaling group status

2. Root Cause Analysis:
   - Load test with production traffic replay
   - Profile model inference with PyInstrument
   - Check for feature computation bottlenecks
   - Review recent deployment changes

3. Common Culprits:
   - Cold start issues with large models
   - Feature store latency increases
   - Memory leaks in custom inference code
   - Network connectivity issues to dependencies
```
