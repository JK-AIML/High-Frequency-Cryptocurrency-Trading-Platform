"""
Data validation framework with support for custom validation rules and quality checks.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Callable, Optional, Any, Union, Tuple
from dataclasses import dataclass, field
from enum import Enum, auto
import logging
from datetime import datetime
import json

logger = logging.getLogger(__name__)

class ValidationLevel(Enum):
    WARNING = auto()
    ERROR = auto()

@dataclass
class ValidationResult:
    """Container for validation results."""
    is_valid: bool
    message: str
    level: ValidationLevel = ValidationLevel.ERROR
    details: Dict[str, Any] = field(default_factory=dict)

class DataValidator:
    """
    Data validation framework for performing quality checks on financial data.
    
    Features:
    - Schema validation
    - Range and boundary checks
    - Custom validation rules
    - Null and missing value detection
    - Statistical validation
    - Cross-field validation
    """
    
    def __init__(self, schema: Optional[Dict] = None):
        """
        Initialize the validator with an optional schema.
        
        Args:
            schema: Dictionary defining expected data types and constraints
        """
        self.schema = schema or {}
        self._validators = {}
        self._register_default_validators()
    
    def _register_default_validators(self) -> None:
        """Register default validation functions."""
        self.register_validator('not_null', self._validate_not_null)
        self.register_validator('data_type', self._validate_data_type)
        self.register_validator('in_range', self._validate_range)
        self.register_validator('unique', self._validate_unique)
        self.register_validator('regex', self._validate_regex)
        self.register_validator('custom', self._validate_custom)
    
    def register_validator(self, name: str, validator: Callable) -> None:
        """Register a custom validation function."""
        self._validators[name] = validator
    
    def validate_schema(self, df: pd.DataFrame) -> List[ValidationResult]:
        """Validate DataFrame against the defined schema."""
        results = []
        
        # Check for missing columns
        missing_columns = set(self.schema.keys()) - set(df.columns)
        if missing_columns:
            results.append(ValidationResult(
                is_valid=False,
                message=f"Missing required columns: {missing_columns}",
                level=ValidationLevel.ERROR
            ))
        
        # Validate each column against its schema
        for col, rules in self.schema.items():
            if col not in df.columns:
                continue
                
            for rule_name, rule_value in rules.items():
                if rule_name in self._validators:
                    result = self._validators[rule_name](df[col], rule_value, col)
                    if not result.is_valid:
                        results.append(result)
        
        return results
    
    def validate_dataframe(self, df: pd.DataFrame, rules: Dict) -> Dict[str, List[ValidationResult]]:
        """
        Validate DataFrame using provided rules.
        
        Args:
            df: Input DataFrame
            rules: Dictionary of validation rules
            
        Returns:
            Dictionary mapping rule names to validation results
        """
        results = {}
        
        for col, col_rules in rules.items():
            if col not in df.columns:
                results[col] = [ValidationResult(
                    is_valid=False,
                    message=f"Column '{col}' not found in DataFrame",
                    level=ValidationLevel.ERROR
                )]
                continue
                
            col_results = []
            for rule_name, rule_value in col_rules.items():
                if rule_name in self._validators:
                    result = self._validators[rule_name](df[col], rule_value, col)
                    col_results.append(result)
            
            if col_results:
                results[col] = col_results
        
        return results
    
    # Built-in validators
    def _validate_not_null(self, series: pd.Series, required: bool, col: str) -> ValidationResult:
        if not required:
            return ValidationResult(True, "")
            
        null_count = series.isnull().sum()
        if null_count > 0:
            return ValidationResult(
                False,
                f"Column '{col}' has {null_count} null values",
                ValidationLevel.ERROR,
                {'null_count': null_count}
            )
        return ValidationResult(True, "")
    
    def _validate_data_type(self, series: pd.Series, expected_type: str, col: str) -> ValidationResult:
        # Handle pandas nullable types
        if pd.api.types.is_integer_dtype(series.dtype):
            actual_type = 'integer'
        elif pd.api.types.is_float_dtype(series.dtype):
            actual_type = 'float'
        elif pd.api.types.is_bool_dtype(series.dtype):
            actual_type = 'boolean'
        elif pd.api.types.is_datetime64_any_dtype(series.dtype):
            actual_type = 'datetime'
        else:
            actual_type = 'string'
            
        if actual_type != expected_type:
            return ValidationResult(
                False,
                f"Column '{col}' has type '{actual_type}' but expected '{expected_type}'",
                ValidationLevel.ERROR
            )
        return ValidationResult(True, "")
    
    def _validate_range(self, series: pd.Series, value_range: tuple, col: str) -> ValidationResult:
        if pd.api.types.is_numeric_dtype(series.dtype):
            min_val, max_val = value_range
            out_of_range = (series < min_val) | (series > max_val)
            count = out_of_range.sum()
            
            if count > 0:
                return ValidationResult(
                    False,
                    f"Column '{col}' has {count} values outside range {value_range}",
                    ValidationLevel.ERROR,
                    {'out_of_range_count': count, 'range': value_range}
                )
        return ValidationResult(True, "")
    
    def _validate_unique(self, series: pd.Series, unique: bool, col: str) -> ValidationResult:
        if not unique:
            return ValidationResult(True, "")
            
        duplicates = series.duplicated()
        duplicate_count = duplicates.sum()
        
        if duplicate_count > 0:
            return ValidationResult(
                False,
                f"Column '{col}' has {duplicate_count} duplicate values",
                ValidationLevel.ERROR,
                {'duplicate_count': duplicate_count}
            )
        return ValidationResult(True, "")
    
    def _validate_regex(self, series: pd.Series, pattern: str, col: str) -> ValidationResult:
        if not pd.api.types.is_string_dtype(series.dtype):
            return ValidationResult(True, "")
            
        invalid = ~series.str.match(pattern, na=False)
        invalid_count = invalid.sum()
        
        if invalid_count > 0:
            return ValidationResult(
                False,
                f"Column '{col}' has {invalid_count} values that don't match pattern '{pattern}'",
                ValidationLevel.ERROR,
                {'invalid_count': invalid_count, 'pattern': pattern}
            )
        return ValidationResult(True, "")
    
    def _validate_custom(self, series: pd.Series, func: Callable, col: str) -> ValidationResult:
        try:
            result = func(series)
            if isinstance(result, bool):
                if not result:
                    return ValidationResult(False, f"Custom validation failed for column '{col}'", ValidationLevel.ERROR)
                return ValidationResult(True, "")
            return result
        except Exception as e:
            return ValidationResult(
                False,
                f"Error in custom validation for column '{col}': {str(e)}",
                ValidationLevel.ERROR
            )
    
    @staticmethod
    def get_default_schema() -> Dict:
        """Get a default schema for financial tick data."""
        return {
            'timestamp': {
                'data_type': 'datetime',
                'not_null': True,
                'unique': True
            },
            'symbol': {
                'data_type': 'string',
                'not_null': True
            },
            'price': {
                'data_type': 'float',
                'not_null': True,
                'in_range': (0, float('inf'))
            },
            'size': {
                'data_type': 'float',
                'not_null': True,
                'in_range': (0, float('inf'))
            },
            'exchange': {
                'data_type': 'string',
                'not_null': False
            },
            'condition': {
                'data_type': 'string',
                'not_null': False
            }
        }
