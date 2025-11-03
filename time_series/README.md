# Demand Forecasting MLOps Pipeline

![Python](https://img.shields.io/badge/python-3.8%20%7C%203.9%20%7C%203.10-blue)
![License](https://img.shields.io/badge/license-MIT-green)

A modular, production-ready MLOps system for retail demand forecasting. This project demonstrates how to build, validate, and deploy time series forecasting models using traditional statistical methods and machine learning, with a focus on reproducibility and scalability.

## 🎯 Business Problem

Retailers face significant financial losses due to:
- **Stockouts**: Lost sales and customer dissatisfaction when products are unavailable
- **Obsolete Stock**: Capital tied up in slow-moving inventory

This system helps maintain optimal inventory levels by accurately forecasting demand, targeting a 98% service level while minimizing dead stock.

## 🏗️ Project Architecture

```
time_series/
├── 📁 data/               # Data storage
│   ├── raw/              # Original, immutable data
│   └── processed/        # Cleaned and feature-engineered data
├── 📁 notebooks/         # Exploration and analysis (Jupyter)
├── 📁 src/               # Source code
│   ├── data/            # Data loading and preprocessing
│   ├── features/        # Feature engineering
│   ├── models/          # Model training and prediction
│   └── utils/           # Configuration and visualization
├── 📁 config/           # Configuration files
├── 📁 tests/            # Comprehensive test suite
├── 📁 models/           # Trained model artifacts
├── 📁 plots/            # Generated visualizations
└── 📁 .github/          # CI/CD workflows
```

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- Git

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/jonatanmendez29/pruebasML.git
   cd pruebasML
   cd time_series
   ```

2. **Create and activate virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   pip install -r requirements-test.txt  # For testing
   ```

4. **Install the package in development mode**
   ```bash
   pip install -e .
   ```

5. **Generate sample data**
   ```bash
   python -c "from src.data.make_dataset import generate_retail_demand_data; generate_retail_demand_data()"
   ```

### Basic Usage

1. **Train a model for a product**
   ```bash
   python -m src.models.train
   ```

2. **Generate predictions**
   ```bash
   python -m src.models.predict
   ```

3. **Run the complete pipeline**
   ```bash
   python run_pipeline.py
   ```

## 📊 Features

### Data Simulation
- Realistic retail data with trends, seasonality, and promotions
- Multiple product categories with different demand patterns
- Holiday effects and promotional campaigns

### Forecasting Methods
- **Traditional Models**: ARIMA, Exponential Smoothing with proper validation
- **Machine Learning**: Random Forest with feature engineering
- **Feature Engineering**: Lag features, rolling statistics, temporal encoding

### MLOps Capabilities
- Modular, reproducible code structure
- Comprehensive testing suite
- Configuration management
- Model versioning and tracking
- Ready for cloud deployment

## 🛠️ Development

### Running Tests

```bash
# Run all tests
python run_tests.py

# Run with coverage
python run_tests.py --coverage

# Run specific test categories
python run_tests.py --unit-only
python run_tests.py --integration-only

# Run tests in parallel
python run_tests.py --parallel
```

### Code Quality

```bash
# Format code
black src/ tests/

# Check code style
flake8 src/ tests/

# Type checking
mypy src/
```

### Project Structure Details

- **`src/data/`**: Data loading, preprocessing, and validation
- **`src/features/`**: Feature engineering pipelines
- **`src/models/`**: Model training, evaluation, and prediction
- **`src/utils/`**: Configuration management and visualization
- **`tests/`**: Unit tests, integration tests, and test fixtures
- **`config/`**: YAML configuration files for easy parameter tuning

## ⚙️ Configuration

The project uses YAML configuration for easy parameter management:

```yaml
# config/parameters.yaml
data:
  raw_data_path: "data/raw/retail_demand_dataset.csv"
  processed_path: "data/processed/"

features:
  lag_periods: [1, 7, 14, 28]
  window_sizes: [7, 28]
  cyclical_features: ['day_of_week', 'month']

model:
  name: "random_forest"
  hyperparameters:
    n_estimators: 100
    max_depth: 15

training:
  target_column: "units_sold"
  validation_split: "2024-01-01"
```

## 📈 Model Performance

The system provides comprehensive model evaluation:

- **Metrics**: MAE, RMSE, MAPE, Cross-validation scores
- **Visualizations**: Actual vs predicted, feature importance, residual analysis
- **Validation**: Time-series cross-validation, statistical tests

## 🚀 Deployment

### Local Deployment

1. **Train models for all products**
   ```bash
   python scripts/train_all_models.py
   ```

2. **Deploy prediction API**
   ```bash
   python api/app.py
   ```

### Cloud Deployment (AWS)

The project is structured for easy cloud migration:

```python
# Future: Cloud deployment ready
# - S3 for data storage
# - SageMaker for model training
# - Lambda for serverless inference
# - Step Functions for pipeline orchestration
```

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/your-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/your-feature`)
5. Open a Pull Request

### Development Setup

```bash
# Install pre-commit hooks
pre-commit install

# Run validation
pre-commit run --all-files
```

## 📚 Learning Resources

This project is part of an educational series on MLOps for Time Series:

1. **Article 1**: [The Multi-Million Dollar Problem - Business Context](https://jonatanmendez.netlify.app/posts/the-multi-million-dollar-problem-stockouts-and-obsolete-stock/)
2. **Article 2**: [Traditional Time Series Models - Statistical Foundations](https://jonatanmendez.netlify.app/posts/our-first-forecast-traditional-time-series-models-on-your-laptop/)
3. **Article 3**: [Machine Learning for Forecasting - Feature Engineering](https://jonatanmendez.netlify.app/posts/beyond-tradition-harnessing-machine-learning-for-demand-forecasting/)
4. **Article 4**: [MLOps Foundations - Project Structure & Reproducibility](https://jonatanmendez.netlify.app/posts/the-mlops-foundation-structuring-our-project-for-reproducibility-and-collaboration/)
5. **Article 5**: [Hitting a Wall - Why Your Laptop Isn't Enough](https://jonatanmendez.netlify.app/posts/hitting-a-wall-why-your-laptop-isnt-enough/)
6. **Article 6**: [Scaling to Cloud - AWS Infrastructure](https://jonatanmendez.netlify.app/posts/cloud-power-up-building-a-scalable-forecast-engine-on-aws/)
7. **Article 7**: [Production Deployment - Monitoring & Maintenance](https://jonatanmendez.netlify.app/posts/the-final-mile-deployment-monitoring-and-business-impact/)

## 🐛 Troubleshooting

Common issues and solutions:

**Issue**: Import errors
**Solution**: Ensure you've installed the package in development mode: `pip install -e .`

**Issue**: Missing data files
**Solution**: Run the data generation script first

**Issue**: Test failures
**Solution**: Check that all dependencies are installed and data is available

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Built with best practices from the MLOps community
- Inspired by real-world retail forecasting challenges
- Uses popular Python data science libraries (pandas, scikit-learn, statsmodels)

## 📞 Support

For support and questions:
- Open an [issue](https://github.com/jonatanmendez29/pruebasML/issues)
- Check the [documentation](docs/)

---

**Ready to forecast?** Start with the [quick start guide](#-quick-start) or explore the [detailed documentation](docs/).
