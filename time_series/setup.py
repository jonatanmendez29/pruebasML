from setuptools import setup, find_packages

with open("requirements.txt") as f:
    requirements = f.read().splitlines()

with open("README.md") as f:
    long_description = f.read()

setup(
    name="demand_forecast",
    version="0.1.0",
    description="A modular MLOps system for retail demand forecasting",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Jonatan Mendez",
    author_email="jonatanmendez29@gmail.com",
    packages=find_packages(where="."),
    package_dir={"": "."},
    package_data={
        "src": ["*.py"],
        "config": ["*.yaml"],
    },
    include_package_data=True,
    install_requires=requirements,
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.8",
    entry_points={
        "console_scripts": [
            "forecast-train=src.models.train:train_model",
            "forecast-predict=src.models.predict:predict_future",
        ],
    },
    keywords="demand forecasting, time series, machine learning, MLOps, retail",
    project_urls={
        "Source": "https://github.com/jonatanmendez29/pruebasML",
        "Bug Reports": "https://github.com/jonatanmendez29/pruebasML/issues",
    },
)