import boto3
from datetime import datetime, timedelta
from typing import Dict


class CostOptimizer:
    """Monitor and optimize cloud costs"""

    def __init__(self):
        self.cloudwatch = boto3.client('cloudwatch')
        self.ce = boto3.client('ce')  # Cost Explorer

    def get_ml_costs(self, start_date: str, end_date: str) -> Dict[str, float]:
        """Get ML-related costs from Cost Explorer"""

        response = self.ce.get_cost_and_usage(
            TimePeriod={
                'Start': start_date,
                'End': end_date
            },
            Granularity='DAILY',
            Metrics=['UnblendedCost'],
            Filter={
                'Dimensions': {
                    'Key': 'SERVICE',
                    'Values': ['Amazon SageMaker', 'AWS Lambda', 'Amazon S3']
                }
            }
        )

        total_cost = 0.0
        daily_costs = {}

        for day in response['ResultsByTime']:
            cost = float(day['Total']['UnblendedCost']['Amount'])
            total_cost += cost
            daily_costs[day['TimePeriod']['Start']] = cost

        return {
            'total_cost': total_cost,
            'daily_costs': daily_costs,
            'average_daily_cost': total_cost / len(daily_costs) if daily_costs else 0
        }

    def calculate_roi(self, business_savings: float, ml_costs: float) -> Dict[str, float]:
        """Calculate ROI of the forecasting system"""

        roi = (business_savings - ml_costs) / ml_costs * 100
        payback_period = ml_costs / (business_savings / 30)  # Months to payback

        return {
            'roi_percentage': roi,
            'payback_period_months': payback_period,
            'net_savings': business_savings - ml_costs,
            'cost_benefit_ratio': business_savings / ml_costs
        }

    def optimize_resources(self, usage_metrics: Dict[str, float]):
        """Suggest resource optimization based on usage"""

        recommendations = []

        # SageMaker instance optimization
        if usage_metrics.get('inference_cpu_utilization', 0) < 30:
            recommendations.append({
                'service': 'SageMaker',
                'recommendation': 'Downsize inference instances',
                'estimated_savings': '40%',
                'risk': 'Low'
            })

        # S3 storage optimization
        if usage_metrics.get('old_model_storage_gb', 0) > 100:
            recommendations.append({
                'service': 'S3',
                'recommendation': 'Implement lifecycle policies for old models',
                'estimated_savings': '60%',
                'risk': 'Low'
            })

        return recommendations


# Cost monitoring function
def cost_monitoring_handler(event, context):
    """Regular cost monitoring and optimization"""

    optimizer = CostOptimizer()

    # Get costs for last 30 days
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')

    costs = optimizer.get_ml_costs(start_date, end_date)

    # Calculate ROI (using estimated business savings)
    # In reality, this would come from your business metrics
    estimated_savings = 50000  # Estimated from reduced stockouts
    roi_analysis = optimizer.calculate_roi(estimated_savings, costs['total_cost'])

    # Generate optimization recommendations
    recommendations = optimizer.optimize_resources({
        'inference_cpu_utilization': 25,  # Would come from CloudWatch
        'old_model_storage_gb': 150  # Would come from S3 inventory
    })

    return {
        'costs': costs,
        'roi_analysis': roi_analysis,
        'recommendations': recommendations
    }
