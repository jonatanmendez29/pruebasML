import boto3
import json
from typing import List
import pandas as pd

from src.cloud.s3_data_manager import S3DataManager


class TrainingOrchestrator:
    """Orchestrate parallel model training across all products"""

    def __init__(self, s3_bucket: str, role_arn: str):
        self.sfn = boto3.client('stepfunctions')
        self.sagemaker = boto3.client('sagemaker')
        self.s3_bucket = s3_bucket
        self.role_arn = role_arn

    def create_training_state_machine(self) -> str:
        """Create Step Function state machine for parallel training"""

        state_machine_definition = {
            "Comment": "Parallel Product Training Pipeline",
            "StartAt": "GetProductList",
            "States": {
                "GetProductList": {
                    "Type": "Task",
                    "Resource": "arn:aws:states:::lambda:invoke",
                    "Parameters": {
                        "FunctionName": "get-products-function",
                        "Payload": {
                            "bucket": self.s3_bucket
                        }
                    },
                    "ResultPath": "$.products",
                    "Next": "MapProducts"
                },
                "MapProducts": {
                    "Type": "Map",
                    "ItemsPath": "$.products",
                    "MaxConcurrency": 50,  # Process 50 products in parallel
                    "Iterator": {
                        "StartAt": "TrainProductModel",
                        "States": {
                            "TrainProductModel": {
                                "Type": "Task",
                                "Resource": "arn:aws:states:::sagemaker:createTrainingJob.sync",
                                "Parameters": {
                                    "TrainingJobName.$": "States.Format('training-{}-{}', $$.Execution.Name, $.product_id')",
                                    "RoleArn": self.role_arn,
                                    "AlgorithmSpecification": {
                                        "TrainingImage": "your-training-image-uri",  # Your custom container
                                        "TrainingInputMode": "File"
                                    },
                                    "InputDataConfig": [
                                        {
                                            "ChannelName": "training",
                                            "DataSource": {
                                                "S3DataSource": {
                                                    "S3DataType": "S3Prefix",
                                                    "S3Uri.$": f"s3://{self.s3_bucket}/demand-forecast/processed/",
                                                    "S3DataDistributionType": "FullyReplicated"
                                                }
                                            }
                                        }
                                    ],
                                    "OutputDataConfig": {
                                        "S3OutputPath": f"s3://{self.s3_bucket}/demand-forecast/models/"
                                    },
                                    "ResourceConfig": {
                                        "InstanceType": "ml.m5.large",
                                        "InstanceCount": 1,
                                        "VolumeSizeInGB": 30
                                    },
                                    "StoppingCondition": {
                                        "MaxRuntimeInSeconds": 3600
                                    },
                                    "HyperParameters": {
                                        "product-id.$": "$.product_id"
                                    }
                                },
                                "End": True
                            }
                        }
                    },
                    "Next": "TrainingComplete"
                },
                "TrainingComplete": {
                    "Type": "Pass",
                    "Result": "All training jobs completed",
                    "End": True
                }
            }
        }

        response = self.sfn.create_state_machine(
            name='ProductTrainingPipeline',
            definition=json.dumps(state_machine_definition),
            roleArn=self.role_arn
        )

        return response['stateMachineArn']

    def start_training_pipeline(self, product_ids: List[str] = None):
        """Start the parallel training pipeline"""

        if product_ids is None:
            # Get all products from S3
            s3_manager = S3DataManager(self.s3_bucket)
            product_ids = s3_manager.list_products()

        execution_input = {
            "products": [{"product_id": pid} for pid in product_ids[:100]]  # Limit for demo
        }

        response = self.sfn.start_execution(
            stateMachineArn=self.state_machine_arn,
            name=f"TrainingExecution-{pd.Timestamp.now().strftime('%Y%m%d-%H%M%S')}",
            input=json.dumps(execution_input)
        )

        return response['executionArn']