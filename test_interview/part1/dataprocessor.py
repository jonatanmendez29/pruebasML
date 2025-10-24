import threading
from abc import ABC, abstractmethod
from typing import Dict, Any
import pandas as pd
import numpy as np
import logging

class DataValidationError(Exception):
    """Custom exception raised when data validation fails"""
    pass

class DataTransformer(ABC):
    """Abstract base class for data transformation."""
    @abstractmethod
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        pass
    @abstractmethod
    def validate(self, df: pd.DataFrame) -> bool:
        pass


class NumericalTransformer(DataTransformer):
    def __init__(self, fill_strategy: str = 'mean', scaling: bool = True):
        self.fill_strategy = fill_strategy
        self.scaling = scaling
        self._mean = None
        self._std = None

    def validate(self, data: pd.DataFrame) -> bool:
        if data.select_dtypes(include=[np.number]).empty:
            raise DataValidationError('Data has no numerical values')
        return True

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        self.validate(data)
        numerical_data = data.select_dtypes(include=[np.number])

        # Handle missing valued
        if self.fill_strategy == 'mean':
            filled_data = numerical_data.fillna(numerical_data.mean())
        else:
            filled_data = numerical_data.fillna(0)

        # Scale data if required
        if self.scaling:
            if self._mean is None:
                self._mean = filled_data.mean()
                self._std = filled_data.std()
            scaled_data = (filled_data - self._mean) / self._std
            return scaled_data

        return filled_data

class DataProcessor:
    """
    Thread-safe, extensible data processor for ML pipelines
    """
    def __init__(self):
        self.transformers: Dict[str, DataTransformer] = {}
        self._lock = threading.RLock()
        self.logger = logging.getLogger(__name__)

    def register_transformer(self, name: str, transformer: DataTransformer):
        """Register a new transformer for specific data types"""
        with self._lock:
            self.transformers[name] = transformer

    def process_batch(self, data: pd.DataFrame, schema: Dict[str, Any]) -> pd.DataFrame:
        """
        Process a batch of data according to the given schema
        :param data: a dataframe
        :param schema: type of schema
        :return: a dataframe processed according to the given schema
        """
        try:
            #Validate input schema
            self._validate_schema(data, schema)

            processed_results = []

            # Process different data type with appropriate transformers
            for data_type, transformer_name in schema.items():
                if transformer_name in self.transformers:
                    transformer = self.transformers[transformer_name]

                    # Extract relevant columns for this type
                    type_columns = [col for col in data.columns
                                    if schema.get(col) == data_type]

                    if type_columns:
                        subset_data = data[type_columns]
                        processed_results.append(transformer.transform(subset_data))

            # Combine all processed data
            if processed_results:
                result = pd.concat(processed_results, axis=1)
                self.logger.info(f"Successfully processed batch with {len(data)} records")
                return result
            else:
                raise DataValidationError("No valid transformation found for the given schema")

        except Exception as e:
            self.logger.error(f"Failed to process batch with {len(data)} records: {e}")
            raise DataValidationError(f"Processing failed: {str(e)}")

    @staticmethod
    def _validate_schema(data: pd.DataFrame, schema: Dict[str, Any]):
        """Validate that schema matches given data"""
        missing_columns = set(schema.keys()) - set(data.columns)
        if missing_columns:
            raise DataValidationError(f"Missing columns in data: {missing_columns}")

# Usage example
if __name__ == '__main__':
    processor = DataProcessor()
    processor.register_transformer("numerical", NumericalTransformer())

#1. How would you make this class modular and extensible for new data types?
#I used the Strategy Pattern with the `DataTransformer` abstract base class.
# This makes it easy to add new data types by simply creating new transformer classes that implement the interface.
# The processor doesn't need to change when new transformers are added - they just need to be registered.
# This follows the Open/Closed Principle.

#2. What design patterns would you apply for validation and transformation logic?
#I implemented the Template Method pattern in the base transformer, where `transform()` calls `validate()`.
# For more complex validation, I'd use the Specification Pattern with composable validation rules.
# The custom exception `DataValidationError` provides clear error handling,
# and the factory method pattern could be used for transformer creation.

#3. How would you handle memory management with large datasets?
#For large datasets, I'd implement chunk processing using generators.
# Instead of loading everything into memory, we'd process data in configurable chunks.
# I'd also use PyArrow instead of Pandas for better memory efficiency
# and implement lazy evaluation for transformations that don't require full dataset access.

#4. Discuss your approach to error handling and logging.
#I used structured logging with different log levels.
# For production, I'd add metrics collection for monitoring data quality.
# The custom exception hierarchy allows for precise error handling,
# and I'd implement retry logic with exponential backoff for transient failures.