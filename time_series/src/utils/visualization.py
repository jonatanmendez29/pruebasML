import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path


def set_plot_style():
    """Set consistent plot style"""
    plt.style.use('seaborn-v0_8')
    sns.set_palette("husl")

    plt.rcParams['figure.figsize'] = (12.0, 6.0)
    plt.rcParams['font.size'] = 12.0
    plt.rcParams['axes.labelsize'] = 14
    plt.rcParams['axes.titlesize'] = 16
    plt.rcParams['xtick.labelsize'] = 12
    plt.rcParams['ytick.labelsize'] = 12


def plot_feature_importance(model, feature_names, product_id, top_n=15):
    """Plot feature importance"""
    set_plot_style()

    if hasattr(model, 'feature_importances_'):
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=True).tail(top_n)

        plt.figure(figsize=(10, 8))
        plt.barh(importance_df['feature'], importance_df['importance'])
        plt.title(f'Top {top_n} Feature Importance - {product_id}')
        plt.xlabel('Importance Score')
        plt.tight_layout()

        # Save plot
        plots_dir = Path('plots')
        plots_dir.mkdir(exist_ok=True)
        plt.savefig(plots_dir / f'feature_importance_{product_id}.png', dpi=300, bbox_inches='tight')
        plt.close()
    else:
        print("Model does not have feature_importances_ attribute")


def plot_prediction_comparison(dates, actual, predicted, product_id):
    """Plot actual vs predicted values"""
    set_plot_style()

    plt.figure(figsize=(14, 8))

    plt.plot(dates, actual, label='Actual Sales', color='black', linewidth=2, alpha=0.8)
    plt.plot(dates, predicted, label='Predicted Sales', color='red', alpha=0.7)

    plt.title(f'Actual vs Predicted Demand - {product_id}')
    plt.xlabel('Date')
    plt.ylabel('Units Sold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()

    # Save plot
    plots_dir = Path('plots')
    plots_dir.mkdir(exist_ok=True)
    plt.savefig(plots_dir / f'predictions_{product_id}.png', dpi=300, bbox_inches='tight')
    plt.close()


def plot_residuals(dates, actual, predicted, product_id):
    """Plot residuals analysis"""
    set_plot_style()

    residuals = actual - predicted

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(f'Residuals Analysis - {product_id}')

    # Residuals over time
    axes[0, 0].plot(dates, residuals, alpha=0.7)
    axes[0, 0].axhline(y=0, color='r', linestyle='--')
    axes[0, 0].set_title('Residuals over Time')
    axes[0, 0].set_xlabel('Date')
    axes[0, 0].set_ylabel('Residuals')
    axes[0, 0].tick_params(axis='x', rotation=45)

    # Distribution of residuals
    axes[0, 1].hist(residuals, bins=30, alpha=0.7, density=True)
    axes[0, 1].set_title('Distribution of Residuals')
    axes[0, 1].set_xlabel('Residuals')
    axes[0, 1].set_ylabel('Density')

    # Q-Q plot
    from scipy import stats
    stats.probplot(residuals, dist="norm", plot=axes[1, 0])
    axes[1, 0].set_title('Q-Q Plot')

    # Residuals vs Predicted
    axes[1, 1].scatter(predicted, residuals, alpha=0.5)
    axes[1, 1].axhline(y=0, color='r', linestyle='--')
    axes[1, 1].set_title('Residuals vs Predicted')
    axes[1, 1].set_xlabel('Predicted Values')
    axes[1, 1].set_ylabel('Residuals')

    plt.tight_layout()

    # Save plot
    plots_dir = Path('plots')
    plots_dir.mkdir(exist_ok=True)
    plt.savefig(plots_dir / f'residuals_{product_id}.png', dpi=300, bbox_inches='tight')
    plt.close()


def plot_time_series_decomposition(df, product_id, date_col='date', value_col='units_sold'):
    """Plot time series decomposition"""
    from statsmodels.tsa.seasonal import seasonal_decompose

    set_plot_style()

    # Prepare data
    product_data = df[df['product_id'] == product_id].sort_values(date_col)
    ts_data = product_data.set_index(date_col)[value_col]

    # Perform decomposition
    decomposition = seasonal_decompose(ts_data, model='additive', period=52)

    # Plot
    fig, axes = plt.subplots(4, 1, figsize=(15, 12))
    fig.suptitle(f'Time Series Decomposition - {product_id}')

    decomposition.observed.plot(ax=axes[0], title='Observed')
    decomposition.trend.plot(ax=axes[1], title='Trend')
    decomposition.seasonal.plot(ax=axes[2], title='Seasonal')
    decomposition.resid.plot(ax=axes[3], title='Residual')

    for ax in axes:
        ax.set_ylabel('Units Sold')

    plt.tight_layout()

    # Save plot
    plots_dir = Path('plots')
    plots_dir.mkdir(exist_ok=True)
    plt.savefig(plots_dir / f'decomposition_{product_id}.png', dpi=300, bbox_inches='tight')
    plt.close()


def plot_correlation_heatmap(df, product_id, columns=None):
    """Plot correlation heatmap for features"""
    set_plot_style()

    if columns is None:
        # Select numerical columns only
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        # Take top 15 most relevant columns to avoid overcrowding
        columns = numerical_cols[:15]

    corr_matrix = df[columns].corr()

    plt.figure(figsize=(12, 10))
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))

    sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f', cmap='coolwarm',
                center=0, square=True, linewidths=0.5)
    plt.title(f'Feature Correlation Heatmap - {product_id}')
    plt.tight_layout()

    # Save plot
    plots_dir = Path('plots')
    plots_dir.mkdir(exist_ok=True)
    plt.savefig(plots_dir / f'correlation_{product_id}.png', dpi=300, bbox_inches='tight')
    plt.close()


if __name__ == "__main__":
    print("Visualization utilities loaded successfully!")