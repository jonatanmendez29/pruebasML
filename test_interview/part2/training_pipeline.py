"""
Architecture Overview:
I'd design a multi-layered system with clear separation between offline and online components.

Data Pipeline:
```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Batch Features│    │ Real-time Features│   │ Feature Store   │
│   (Spark)       │────│ (Kafka/Flink)    │────│ (Redis/DynamoDB)│
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                │
                        ┌───────┴───────┐
                        │ Model Service │
                        │ (SageMaker)   │
                        └───────┬───────┘
                                │
                        ┌───────┴───────┐
                        │   API Layer   │
                        │ (API Gateway) │
                        └───────────────┘
```

Feature Engineering Strategy:

- **Batch Features:** User historical preferences, item popularity trends computed daily using Spark ETL jobs
- **Real-time Features:** User session behavior, recent interactions processed via Kafka streams with Flink
- **Feature Store:** Redis for low-latency feature serving with DynamoDB as backup
- **Schema Management:** Protobuf schemas with versioning to handle evolution
"""

# Pseudo-code for training pipeline
def training_pipeline():
    # Feature retrieval from feature store
    features = feature_store.get_training_features()

    # Hyperparameter optimization
    best_params = bayesian_optimization(
        objective=train_evaluate_model,
        search_space=hp_space
    )

    # Model training with cross-validation
    model = train_final_model(features, best_params)

    # Model validation against current champion
    metrics = validate_model(model, test_data)

    if metrics['ndcg'] > champion_metrics['ndcg'] * 1.05:  # 5% improvement
        model_registry.register_model(model, version='candidate')
        trigger_canary_deployment()

"""
**A/B Testing Strategy:**
- **Canary Deployment:** 5% traffic initially, gradually increasing based on performance
- **Multi-armed Bandit:** For dynamic traffic allocation between model versions
- **Metrics:** Primary - conversion rate; Secondary - click-through rate, session duration
- **Statistical Significance:** Use sequential testing with early stopping rules

**Monitoring & Drift Detection:**
- **Data Drift:** Monitor feature distributions using PSI (Population Stability Index)
- **Concept Drift:** Track prediction distribution shifts and performance degradation
- **Business Metrics:** Real-time dashboard for recommendation engagement metrics
- **Alerting:** Automated alerts when metrics cross predefined thresholds

**AWS Service Selection Rationale:**
- **SageMaker:** For managed training and deployment with built-in A/B testing
- **EMR/Glue:** For large-scale batch feature processing
- **Kinesis/MSK:** For real-time feature streaming
- **Redis Elasticache:** For low-latency feature serving
- **CloudWatch:** For comprehensive monitoring and alerting
"""