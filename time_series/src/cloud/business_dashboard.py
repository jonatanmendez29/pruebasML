import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Dict, List, Any
import json


class BusinessDashboard:
    """Dashboard showing business impact of forecasts"""

    def __init__(self, s3_bucket: str):
        self.s3_bucket = s3_bucket

    def calculate_service_level(self, actual_demand: List[float], inventory: List[float]) -> float:
        """Calculate achieved service level"""
        stockouts = sum(1 for a, i in zip(actual_demand, inventory) if a > i)
        total_periods = len(actual_demand)

        service_level = (1 - stockouts / total_periods) * 100
        return service_level

    def calculate_inventory_turnover(self, sales: List[float], avg_inventory: List[float]) -> float:
        """Calculate inventory turnover ratio"""
        return sum(sales) / sum(avg_inventory)

    def calculate_cost_savings(self, current_stockouts: int, improved_stockouts: int,
                               avg_order_value: float) -> float:
        """Calculate cost savings from improved forecasts"""
        prevented_stockouts = current_stockouts - improved_stockouts
        return prevented_stockouts * avg_order_value

    def generate_performance_report(self, product_id: str, start_date: str, end_date: str) -> Dict[str, Any]:
        """Generate comprehensive business performance report"""

        # Load historical data
        from s3_data_manager import S3DataManager
        s3_mgr = S3DataManager(self.s3_bucket)

        try:
            data = s3_mgr.download_dataframe(f"processed/{product_id}.parquet")
            data = data[(data['date'] >= start_date) & (data['date'] <= end_date)]

            # Simulate inventory decisions based on forecasts
            # In reality, this would come from your inventory system
            data['inventory_level'] = data['predicted_demand'] * 1.1  # 10% buffer
            data['stockout'] = data['units_sold'] > data['inventory_level']

            # Calculate key metrics
            service_level = self.calculate_service_level(
                data['units_sold'].tolist(),
                data['inventory_level'].tolist()
            )

            inventory_turnover = self.calculate_inventory_turnover(
                data['units_sold'].tolist(),
                data['inventory_level'].tolist()
            )

            total_stockouts = data['stockout'].sum()
            potential_revenue_lost = total_stockouts * data['selling_price'].mean()

            return {
                'product_id': product_id,
                'period': f"{start_date} to {end_date}",
                'service_level': round(service_level, 1),
                'inventory_turnover': round(inventory_turnover, 2),
                'total_stockouts': int(total_stockouts),
                'potential_revenue_lost': round(potential_revenue_lost, 2),
                'target_service_level': 98.0,
                'meeting_target': service_level >= 98.0
            }

        except Exception as e:
            return {'error': str(e)}

    def create_dashboard_visualization(self, report_data: Dict[str, Any]):
        """Create interactive dashboard visualization"""

        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Service Level Performance', 'Inventory Turnover',
                            'Stockout Analysis', 'Business Impact'),
            specs=[[{"type": "indicator"}, {"type": "indicator"}],
                   [{"type": "bar"}, {"type": "bar"}]]
        )

        # Service level gauge
        fig.add_trace(
            go.Indicator(
                mode="gauge+number+delta",
                value=report_data['service_level'],
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Service Level %"},
                delta={'reference': 98},
                gauge={'axis': {'range': [None, 100]},
                       'bar': {'color': "darkblue"},
                       'steps': [{'range': [0, 90], 'color': "lightgray"},
                                 {'range': [90, 98], 'color': "yellow"},
                                 {'range': [98, 100], 'color': "green"}]}
            ),
            row=1, col=1
        )

        # Inventory turnover
        fig.add_trace(
            go.Indicator(
                mode="number",
                value=report_data['inventory_turnover'],
                title={'text': "Inventory Turnover"},
            ),
            row=1, col=2
        )

        # Stockout analysis
        fig.add_trace(
            go.Bar(
                x=['Actual Stockouts', 'Target Stockouts'],
                y=[report_data['total_stockouts'], 0],
                marker_color=['red', 'green']
            ),
            row=2, col=1
        )

        # Revenue impact
        fig.add_trace(
            go.Bar(
                x=['Potential Revenue Lost'],
                y=[report_data['potential_revenue_lost']],
                marker_color='orange'
            ),
            row=2, col=2
        )

        fig.update_layout(height=600, showlegend=False,
                          title_text=f"Business Impact Dashboard - {report_data['product_id']}")

        return fig


# API endpoint for business dashboard
def dashboard_api_handler(event, context):
    """API handler for business dashboard"""

    dashboard = BusinessDashboard('your-bucket')

    product_id = event.get('queryStringParameters', {}).get('product_id', 'P003')
    start_date = event.get('queryStringParameters', {}).get('start_date', '2023-01-01')
    end_date = event.get('queryStringParameters', {}).get('end_date', '2023-12-31')

    try:
        report = dashboard.generate_performance_report(product_id, start_date, end_date)

        if 'error' in report:
            return {
                'statusCode': 500,
                'body': json.dumps({'error': report['error']})
            }

        # Generate visualization
        fig = dashboard.create_dashboard_visualization(report)

        return {
            'statusCode': 200,
            'body': json.dumps({
                'report': report,
                'visualization': fig.to_json()  # Would be rendered by frontend
            })
        }

    except Exception as e:
        return {
            'statusCode': 500,
            'body': json.dumps({'error': str(e)})
        }