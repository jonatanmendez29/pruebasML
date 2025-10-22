# Contributing to Demand Forecasting MLOps Pipeline

Thank you for your interest in contributing to the Demand Forecasting MLOps Pipeline! This document provides guidelines and instructions for contributing to this project.

## 🎯 Our Philosophy

We believe in:
- **Reproducibility**: Every result should be reproducible
- **Modularity**: Clean, modular code that's easy to understand and extend
- **Testing**: Comprehensive testing for reliable code
- **Documentation**: Clear documentation for users and contributors
- **Collaboration**: Friendly, inclusive collaboration

## 📋 Table of Contents

- [Code of Conduct](#-code-of-conduct)
- [Getting Started](#-getting-started)
- [Development Workflow](#-development-workflow)
- [Code Standards](#-code-standards)
- [Testing](#-testing)
- [Documentation](#-documentation)
- [Pull Request Process](#-pull-request-process)
- [Issue Reporting](#-issue-reporting)
- [Feature Requests](#-feature-requests)

## 📜 Code of Conduct

### Our Pledge

We as members, contributors, and leaders pledge to make participation in our community a harassment-free experience for everyone, regardless of age, body size, visible or invisible disability, ethnicity, sex characteristics, gender identity and expression, level of experience, education, socio-economic status, nationality, personal appearance, race, religion, or sexual identity and orientation.

### Our Standards

Examples of behavior that contributes to a positive environment:

- Using welcoming and inclusive language
- Being respectful of differing viewpoints and experiences
- Gracefully accepting constructive criticism
- Focusing on what is best for the community
- Showing empathy towards other community members

### Enforcement

Instances of abusive, harassing, or otherwise unacceptable behavior may be reported by contacting the project team. All complaints will be reviewed and investigated promptly and fairly.

## 🚀 Getting Started

### Prerequisites

- Python 3.8 or higher
- Git
- (Recommended) Familiarity with MLOps concepts

### Setting Up Development Environment

1. **Fork the repository**
   ```bash
   # Click the 'Fork' button on GitHub, then:
   git clone https://github.com/jonatanmendez29/pruebasML.git
   cd pruebasML
   cd time_series
   ```

2. **Set up development environment**
   ```bash
   # Create virtual environment
   python -m venv venv
   source venv/bin/activate  # Linux/Mac
   # venv\Scripts\activate  # Windows
   
   # Install development dependencies
   pip install -r requirements.txt
   pip install -r requirements-test.txt
   pip install -e .
   ```

3. **Set up pre-commit hooks**
   ```bash
   pre-commit install
   ```

4. **Generate sample data**
   ```bash
   python -c "from src.data.make_dataset import generate_retail_demand_data; generate_retail_demand_data()"
   ```

## 🔄 Development Workflow

### Branch Naming Convention

Use the following prefixes for your branches:

- `feature/`: New features
- `bugfix/`: Bug fixes
- `hotfix/`: Critical production fixes
- `docs/`: Documentation updates
- `test/`: Test-related changes
- `refactor/`: Code refactoring

Examples:
- `feature/add-prophet-model`
- `bugfix/fix-data-leakage`
- `docs/update-contributing-guide`

### Commit Message Convention

We follow the [Conventional Commits](https://www.conventionalcommits.org/) specification:

```
<type>[optional scope]: <description>

[optional body]

[optional footer(s)]
```

Types:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes (formatting, etc.)
- `refactor`: Code refactoring
- `test`: Test-related changes
- `chore`: Maintenance tasks

Examples:
- `feat: add XGBoost model support`
- `fix: resolve data preprocessing issue`
- `docs: update installation instructions`

## 🛠️ Code Standards

### Python Code Style

We use:
- **Black** for code formatting
- **Flake8** for linting
- **isort** for import sorting
- **mypy** for type checking

Run code quality checks:
```bash
# Format code
black src/ tests/

# Sort imports
isort src/ tests/

# Check code style
flake8 src/ tests/

# Type checking
mypy src/
```

### Python Guidelines

1. **Use type hints** for all function signatures
   ```python
   def preprocess_data(df: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
   ```

2. **Write docstrings** for all public functions and classes
   ```python
   def create_lag_features(df: pd.DataFrame, lag_periods: List[int]) -> pd.DataFrame:
       """
       Create lag features for time series data.
       
       Parameters:
       - df: Input DataFrame with time series data
       - lag_periods: List of lag periods to create
       
       Returns:
       - DataFrame with added lag features
       """
   ```

3. **Follow PEP 8** naming conventions
   - Functions: `snake_case`
   - Classes: `PascalCase`
   - Constants: `UPPER_SNAKE_CASE`

4. **Keep functions focused** and single-responsibility

### Configuration Standards

- Use YAML for configuration files
- Document all configuration parameters
- Provide sensible defaults
- Validate configuration on load

### MLOps Standards

- All data processing must be reproducible
- Models should be versioned and stored with metadata
- Feature engineering must prevent data leakage
- Tests should cover data validation and model performance

## ✅ Testing

### Testing Standards

1. **Write tests for new features**
2. **Maintain test coverage above 80%**
3. **Include both unit tests and integration tests**
4. **Test edge cases and error conditions**

### Running Tests

```bash
# Run all tests
python run_tests.py

# Run specific test categories
python run_tests.py --unit-only
python run_tests.py --integration-only

# Run with coverage
python run_tests.py --coverage

# Run specific test file
pytest tests/test_models.py -v

# Run tests in parallel
python run_tests.py --parallel
```

### Writing Tests

```python
class TestFeatureEngineering:
    """Test feature engineering functionality"""
    
    def test_lag_features_creation(self, sample_product_data):
        """Test creation of lag features"""
        from src.features.lag_features import create_lag_features
        
        df_with_lags = create_lag_features(sample_product_data)
        
        # Check lag features are created
        assert 'lag_1' in df_with_lags.columns
        assert 'lag_7' in df_with_lags.columns
        
        # Check no data leakage
        assert pd.isna(df_with_lags['lag_1'].iloc[0])
```

## 📚 Documentation

### Documentation Standards

1. **Update README.md** for significant changes
2. **Document new configuration parameters**
3. **Add docstrings to all new functions and classes**
4. **Update examples if behavior changes**

### Building Documentation

```bash
# Build documentation locally
cd docs
make html
```

### Inline Documentation

- Use Google-style docstrings
- Include examples for complex functions
- Document parameter types and return values
- Mention exceptions that might be raised

## 🔄 Pull Request Process

### Before Submitting a PR

1. **Ensure your code passes all tests**
   ```bash
   python run_tests.py
   ```

2. **Run code quality checks**
   ```bash
   black src/ tests/
   isort src/ tests/
   flake8 src/ tests/
   mypy src/
   ```

3. **Update documentation** if needed
4. **Add tests** for new functionality
5. **Update the CHANGELOG.md** with your changes

### PR Checklist

- [ ] Tests pass
- [ ] Code follows style guidelines
- [ ] Documentation updated
- [ ] CHANGELOG updated
- [ ] Branch is up to date with main
- [ ] Self-review completed

### PR Description Template

```markdown
## Description
Brief description of the changes

## Related Issue
Fixes #(issue number)

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Documentation update
- [ ] Refactoring
- [ ] Performance improvement

## Testing
Describe the tests you ran to verify your changes

## Screenshots (if applicable)

## Additional Notes
Any additional information reviewers should know
```

## 🐛 Issue Reporting

### Creating Good Bug Reports

1. **Use the issue template**
2. **Provide reproduction steps**
3. **Include error messages and stack traces**
4. **Specify your environment** (OS, Python version, etc.)
5. **Add screenshots** if applicable

### Issue Labels

- `bug`: Something isn't working
- `enhancement`: New feature or improvement
- `documentation`: Documentation issues or improvements
- `question`: Further information is requested
- `good first issue`: Good for newcomers
- `help wanted`: Extra attention is needed

## 💡 Feature Requests

We welcome feature requests! When suggesting a new feature:

1. **Check existing issues** to avoid duplicates
2. **Describe the problem** you're trying to solve
3. **Explain why this feature would help**
4. **Provide examples** of how it would be used
5. **Consider implementation complexity**

### Feature Request Template

```markdown
## Problem Description
What problem are you trying to solve?

## Proposed Solution
How would this feature work?

## Alternatives Considered
What other approaches have you considered?

## Additional Context
Any other information that might be helpful
```

## 🏗️ Project Structure Contributions

### Adding New Models

1. Create model class in `src/models/`
2. Add configuration parameters in `config/parameters.yaml`
3. Write tests in `tests/test_models.py`
4. Update documentation

### Adding New Features

1. Create feature engineering functions in `src/features/`
2. Add configuration parameters
3. Write tests
4. Update documentation

### Adding New Data Sources

1. Create data loader in `src/data/`
2. Add data validation
3. Update preprocessing if needed
4. Write tests

## 🎖️ Recognition

Contributors will be recognized in:
- GitHub contributors list
- Release notes
- Project documentation

## ❓ Getting Help

- **Documentation**: Check the README and code docstrings
- **Issues**: Search existing issues
- **Discussions**: Use GitHub Discussions for questions
- **Contact**: Reach out to maintainers for direct help

## 📝 Changelog Updates

When making changes, update `CHANGELOG.md` following [Keep a Changelog](https://keepachangelog.com/) format:

```markdown
## [Unreleased]

### Added
- New features

### Changed
- Existing functionality

### Fixed
- Bug fixes
```

## 🔍 Review Process

1. **Initial review** within 48 hours
2. **Feedback provided** within 3-5 days
3. **Iteration** until all concerns addressed
4. **Merge** by maintainers

## 🏆 Good First Issues

Look for issues labeled `good first issue` if you're new to the project. These are specifically chosen for newcomers.

## 🤝 Community

Join our community:
- GitHub Discussions for questions
- Contributor spotlight in releases

---

Thank you for contributing to making this project better! Your efforts help the entire MLOps community.

*Last updated: 10-22-2015*