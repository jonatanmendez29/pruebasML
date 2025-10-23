import pandas as pd
import time
import psutil
import warnings
warnings.filterwarnings('ignore')
from pathlib import Path
import sys

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent))
from src.models.train import train_model

def attempt_scale_simulation():
    """Simulate what happens when we try to scale locally"""

    # Simulate multiple products
    product_ids = [f'P{str(i).zfill(3)}' for i in range(1, 101)]  # Just 100 products

    results = []
    memory_usage = []

    for i, product_id in enumerate(product_ids):  # Try first 10
        try:
            # Monitor resources
            memory_before = psutil.virtual_memory().percent

            # Time the training
            start_time = time.time()

            # This will fail for most products without their specific data
            # But we're simulating the attempt
            print(f"Processing {product_id}...")
            model, feature_columns, metrics = train_model(product_id, save_model=False)

            training_time = time.time() - start_time
            memory_after = psutil.virtual_memory().percent

            results.append({
                'product_id': product_id,
                'training_time': training_time,
                'memory_increase': memory_after - memory_before,
                'status': 'success'
            })

            memory_usage.append(memory_after)

            # Check if we're running out of resources
            if memory_after > 85:
                print("⚠️  Memory usage critical - stopping simulation")
                break

        except Exception as e:
            results.append({
                'product_id': product_id,
                'training_time': 0,
                'memory_increase': 0,
                'status': f'failed: {str(e)}'
            })

    return pd.DataFrame(results)

# Run the simulation
print("🚨 Attempting to scale locally...")
results_df = attempt_scale_simulation()
print("\nResults:")
print(results_df.head())

if len(results_df) > 0:
    avg_time = results_df[results_df['status'] == 'success']['training_time'].mean()
    total_estimated = avg_time * 10000 / 3600  # Estimate for 10K products
    print(f"\n📈 Estimated time for 10,000 products: {total_estimated:.1f} hours")