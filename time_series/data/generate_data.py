import pandas as pd
import numpy as np
from datetime import datetime, timedelta


def generate_retail_demand_data(start_date='2022-01-01', end_date='2024-12-31', random_seed=42):
    """
    Generate realistic retail demand data for multiple products
    """
    np.random.seed(random_seed)

    # Create date range
    dates = pd.date_range(start=start_date, end=end_date, freq='D')
    n_days = len(dates)

    # Product catalog
    products = [
        {'product_id': 'P001', 'category': 'Electronics', 'base_price': 299.99, 'base_demand': 5, 'trend': 'growth'},
        {'product_id': 'P002', 'category': 'Electronics', 'base_price': 89.99, 'base_demand': 8, 'trend': 'stable'},
        {'product_id': 'P003', 'category': 'Clothing', 'base_price': 49.99, 'base_demand': 40, 'trend': 'seasonal'},
        {'product_id': 'P004', 'category': 'Clothing', 'base_price': 79.99, 'base_demand': 30, 'trend': 'decline'},
        {'product_id': 'P005', 'category': 'Home', 'base_price': 149.99, 'base_demand': 20, 'trend': 'stable'},
        {'product_id': 'P006', 'category': 'Home', 'base_price': 39.99, 'base_demand': 50, 'trend': 'growth'},
    ]

    all_data = []

    for product in products:
        product_data = []

        for i, date in enumerate(dates):
            # Base demand
            demand = product['base_demand']

            # 1. TREND COMPONENT
            if product['trend'] == 'growth':
                trend_factor = 1 + (i / n_days) * 0.8  # 80% growth over period
            elif product['trend'] == 'decline':
                trend_factor = 1 - (i / n_days) * 0.5  # 50% decline over period
            else:  # stable
                trend_factor = 1 + np.random.normal(0, 0.1)

            demand *= trend_factor

            # 2. WEEKLY SEASONALITY
            day_of_week = date.dayofweek
            if day_of_week in [5, 6]:  # Weekend boost
                if product['category'] == 'Clothing':
                    demand *= 1.8  # Big weekend boost for clothing
                else:
                    demand *= 1.4
            else:
                demand *= 0.9  # Weekday dip

            # 3. YEARLY SEASONALITY
            day_of_year = date.timetuple().tm_yday

            if product['category'] == 'Clothing':
                # Seasonal clothing (winter and summer peaks)
                if 300 <= day_of_year <= 365 or 0 <= day_of_year <= 60:  # Winter
                    demand *= 1.6
                elif 150 <= day_of_year <= 240:  # Summer
                    demand *= 1.4
                else:  # Spring/Fall
                    demand *= 0.7
            elif product['category'] == 'Electronics':
                # Holiday season boost
                if 320 <= day_of_year <= 365:  # Nov-Dec
                    demand *= 2.0
                elif 0 <= day_of_year <= 30:  # Post-holiday returns period
                    demand *= 0.6

            # 4. PROMOTIONS (random but strategic)
            promotion = 0
            # Higher promotion probability during low seasons or holidays
            promo_probability = 0.05  # Base 5% chance
            if day_of_year >= 320:  # Holiday season
                promo_probability = 0.15
            elif product['trend'] == 'decline':
                promo_probability = 0.1

            if np.random.random() < promo_probability:
                promotion = 1
                # Promotion effect: 40-100% demand boost
                promo_boost = 0.4 + np.random.random() * 0.6
                demand *= (1 + promo_boost)

                # Promotion usually lasts 3-7 days
                promo_duration = np.random.randint(3, 8)
                for j in range(1, promo_duration):
                    if i + j < n_days:
                        # Add future promotion days
                        future_date = dates[i + j]
                        product_data.append({
                            'date': future_date,
                            'product_id': product['product_id'],
                            'category': product['category'],
                            'base_price': product['base_price'],
                            'promotion': 1,
                            # We'll calculate demand for these later
                            'units_sold': None
                        })

            # 5. HOLIDAY EFFECTS
            holiday = 0
            # Major US holidays
            major_holidays = [
                (12, 25),  # Christmas
                (12, 31),  # New Year's Eve
                (1, 1),  # New Year's Day
                (7, 4),  # Independence Day
                (11, 28, True),  # Thanksgiving (4th Thursday)
            ]

            for holiday_def in major_holidays:
                if len(holiday_def) == 3:  # Thanksgiving (floating holiday)
                    # Simple approximation - 4th Thursday in November
                    if date.month == 11 and date.weekday() == 3 and 21 <= date.day <= 28:
                        holiday = 1
                        demand *= 1.3
                elif date.month == holiday_def[0] and date.day == holiday_def[1]:
                    holiday = 1
                    if holiday_def == (12, 25):  # Christmas big boost
                        demand *= 1.8
                    else:
                        demand *= 1.4

            # 6. RANDOM NOISE AND OUTLIERS
            noise = np.random.normal(1, 0.2)  # 20% random variation
            demand *= noise

            # Occasional outliers (5% chance)
            if np.random.random() < 0.05:
                outlier_effect = np.random.choice([0.3, 2.5])  # Either very low or very high
                demand *= outlier_effect

            # Ensure minimum demand
            demand = max(1, demand)

            # Calculate selling price (with promotions)
            selling_price = product['base_price']
            if promotion:
                selling_price *= (1 - np.random.uniform(0.1, 0.3))  # 10-30% discount

            product_data.append({
                'date': date,
                'product_id': product['product_id'],
                'category': product['category'],
                'base_price': product['base_price'],
                'selling_price': round(selling_price, 2),
                'promotion': promotion,
                'holiday': holiday,
                'units_sold': int(round(demand))
            })

        # Remove duplicate entries from multi-day promotions
        df_product = pd.DataFrame(product_data).drop_duplicates(subset=['date', 'product_id'])
        all_data.append(df_product)

    # Combine all products
    final_df = pd.concat(all_data, ignore_index=True)

    # Add some derived features
    final_df['year'] = final_df['date'].dt.year
    final_df['month'] = final_df['date'].dt.month
    final_df['day_of_week'] = final_df['date'].dt.dayofweek
    final_df['day_of_year'] = final_df['date'].dt.dayofyear
    final_df['is_weekend'] = (final_df['day_of_week'] >= 5).astype(int)

    # Calculate revenue
    final_df['revenue'] = final_df['units_sold'] * final_df['selling_price']

    # Reorder columns
    cols = ['date', 'product_id', 'category', 'base_price', 'selling_price',
            'promotion', 'holiday', 'units_sold', 'revenue', 'year', 'month',
            'day_of_week', 'day_of_year', 'is_weekend']
    final_df = final_df[cols]

    return final_df.sort_values(['product_id', 'date']).reset_index(drop=True)


# Generate and save the dataset
print("Generating retail demand dataset...")
df = generate_retail_demand_data()

print(f"Dataset shape: {df.shape}")
print(f"Date range: {df['date'].min()} to {df['date'].max()}")
print(f"Products: {df['product_id'].nunique()}")
print(f"Total records: {len(df)}")
print("\nFirst few rows:")
print(df.head(10))

print("\nSample statistics by product:")
print(df.groupby('product_id').agg({
    'units_sold': ['mean', 'std', 'min', 'max'],
    'revenue': 'sum',
    'promotion': 'mean'
}).round(2))

# Save to CSV
df.to_csv('retail_demand_dataset.csv', index=False)
print("\nDataset saved as 'retail_demand_dataset.csv'")