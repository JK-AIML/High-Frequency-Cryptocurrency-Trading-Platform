"""
Enhanced Data Validation Module

This module provides comprehensive data validation capabilities including
schema validation, type validation, range validation, and custom validation rules.
"""

import logging
from typing import Dict, List, Any, Optional, Union, Callable
from dataclasses import dataclass
from datetime import datetime
import json
import jsonschema
from pydantic import BaseModel, ValidationError as PydanticValidationError
import numpy as np
from enum import Enum

logger = logging.getLogger(__name__)

class ValidationSeverity(Enum):
    """Severity levels for validation errors."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"

@dataclass
class ValidationResult:
    """Result of a validation check."""
    is_valid: bool
    errors: List[Dict[str, Any]]
    warnings: List[Dict[str, Any]]
    metadata: Dict[str, Any]

class SchemaValidator:
    """Validates data against a JSON schema."""
    
    def __init__(self, schema: Dict[str, Any]):
        """
        Initialize the schema validator.
        
        Args:
            schema: JSON schema to validate against
        """
        self.schema = schema
        self.validator = jsonschema.Draft7Validator(schema)
    
    def validate(self, data: Dict[str, Any]) -> ValidationResult:
        """
        Validate data against the schema.
        
        Args:
            data: Data to validate
            
        Returns:
            ValidationResult containing validation results
        """
        errors = []
        warnings = []
        
        try:
            self.validator.validate(data)
        except jsonschema.exceptions.ValidationError as e:
            errors.append({
                'message': str(e),
                'path': list(e.path),
                'severity': ValidationSeverity.ERROR.value
            })
        
        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            metadata={'schema_version': 'draft-07'}
        )

class TypeValidator:
    """Validates data types using Pydantic models."""
    
    def __init__(self, model_class: type[BaseModel]):
        """
        Initialize the type validator.
        
        Args:
            model_class: Pydantic model class to validate against
        """
        self.model_class = model_class
    
    def validate(self, data: Dict[str, Any]) -> ValidationResult:
        """
        Validate data types.
        
        Args:
            data: Data to validate
            
        Returns:
            ValidationResult containing validation results
        """
        errors = []
        warnings = []
        
        try:
            self.model_class(**data)
        except PydanticValidationError as e:
            for error in e.errors():
                errors.append({
                    'message': error['msg'],
                    'path': error['loc'],
                    'severity': ValidationSeverity.ERROR.value
                })
        
        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            metadata={'model': self.model_class.__name__}
        )

class RangeValidator:
    """Validates numeric ranges and constraints."""
    
    def __init__(self, constraints: Dict[str, Dict[str, Any]]):
        """
        Initialize the range validator.
        
        Args:
            constraints: Dictionary of field constraints
                Example: {
                    'price': {'min': 0, 'max': 1000000},
                    'volume': {'min': 0}
                }
        """
        self.constraints = constraints
    
    def validate(self, data: Dict[str, Any]) -> ValidationResult:
        """
        Validate numeric ranges.
        
        Args:
            data: Data to validate
            
        Returns:
            ValidationResult containing validation results
        """
        errors = []
        warnings = []
        
        for field, constraints in self.constraints.items():
            if field not in data:
                continue
                
            value = data[field]
            if not isinstance(value, (int, float)):
                continue
            
            # Check minimum value
            if 'min' in constraints and value < constraints['min']:
                errors.append({
                    'message': f"{field} value {value} is below minimum {constraints['min']}",
                    'path': [field],
                    'severity': ValidationSeverity.ERROR.value
                })
            
            # Check maximum value
            if 'max' in constraints and value > constraints['max']:
                errors.append({
                    'message': f"{field} value {value} is above maximum {constraints['max']}",
                    'path': [field],
                    'severity': ValidationSeverity.ERROR.value
                })
            
            # Check warning thresholds
            if 'warning_min' in constraints and value < constraints['warning_min']:
                warnings.append({
                    'message': f"{field} value {value} is below warning threshold {constraints['warning_min']}",
                    'path': [field],
                    'severity': ValidationSeverity.WARNING.value
                })
            
            if 'warning_max' in constraints and value > constraints['warning_max']:
                warnings.append({
                    'message': f"{field} value {value} is above warning threshold {constraints['warning_max']}",
                    'path': [field],
                    'severity': ValidationSeverity.WARNING.value
                })
        
        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            metadata={'constraints': self.constraints}
        )

class CustomValidator:
    """Base class for custom validation rules."""
    
    def __init__(self, name: str):
        """
        Initialize the custom validator.
        
        Args:
            name: Name of the validator
        """
        self.name = name
    
    def validate(self, data: Dict[str, Any]) -> ValidationResult:
        """
        Validate data using custom rules.
        
        Args:
            data: Data to validate
            
        Returns:
            ValidationResult containing validation results
        """
        raise NotImplementedError("Custom validators must implement validate()")

class StatisticalValidator(CustomValidator):
    """Validates data using statistical methods."""
    
    def __init__(self, 
                 field: str,
                 mean: float,
                 std: float,
                 z_score_threshold: float = 3.0):
        """
        Initialize the statistical validator.
        
        Args:
            field: Field to validate
            mean: Expected mean value
            std: Expected standard deviation
            z_score_threshold: Threshold for z-score based outliers
        """
        super().__init__(f"statistical_{field}")
        self.field = field
        self.mean = mean
        self.std = std
        self.z_score_threshold = z_score_threshold
    
    def validate(self, data: Dict[str, Any]) -> ValidationResult:
        """
        Validate data using statistical methods.
        
        Args:
            data: Data to validate
            
        Returns:
            ValidationResult containing validation results
        """
        errors = []
        warnings = []
        
        if self.field not in data:
            return ValidationResult(
                is_valid=True,
                errors=errors,
                warnings=warnings,
                metadata={'field': self.field}
            )
        
        value = data[self.field]
        if not isinstance(value, (int, float)):
            return ValidationResult(
                is_valid=True,
                errors=errors,
                warnings=warnings,
                metadata={'field': self.field}
            )
        
        # Calculate z-score
        z_score = abs((value - self.mean) / self.std)
        
        if z_score > self.z_score_threshold:
            errors.append({
                'message': f"{self.field} value {value} is a statistical outlier (z-score: {z_score:.2f})",
                'path': [self.field],
                'severity': ValidationSeverity.ERROR.value
            })
        
        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            metadata={
                'field': self.field,
                'z_score': z_score,
                'mean': self.mean,
                'std': self.std
            }
        )

class EnhancedValidationProcessor:
    """Enhanced validation processor combining multiple validators."""
    
    def __init__(self,
                 schema: Optional[Dict[str, Any]] = None,
                 model_class: Optional[type[BaseModel]] = None,
                 constraints: Optional[Dict[str, Dict[str, Any]]] = None,
                 custom_validators: Optional[List[CustomValidator]] = None):
        """
        Initialize the enhanced validation processor.
        
        Args:
            schema: JSON schema for validation
            model_class: Pydantic model class for type validation
            constraints: Dictionary of field constraints
            custom_validators: List of custom validators
        """
        self.validators = []
        
        if schema:
            self.validators.append(SchemaValidator(schema))
        
        if model_class:
            self.validators.append(TypeValidator(model_class))
        
        if constraints:
            self.validators.append(RangeValidator(constraints))
        
        if custom_validators:
            self.validators.extend(custom_validators)
    
    async def validate(self, data: Dict[str, Any]) -> ValidationResult:
        """
        Validate data using all configured validators.
        
        Args:
            data: Data to validate
            
        Returns:
            ValidationResult containing validation results
        """
        all_errors = []
        all_warnings = []
        all_metadata = {}
        
        for validator in self.validators:
            result = validator.validate(data)
            
            all_errors.extend(result.errors)
            all_warnings.extend(result.warnings)
            all_metadata[validator.__class__.__name__] = result.metadata
        
        return ValidationResult(
            is_valid=len(all_errors) == 0,
            errors=all_errors,
            warnings=all_warnings,
            metadata=all_metadata
        )
    
    async def validate_batch(self, data_batch: List[Dict[str, Any]]) -> List[ValidationResult]:
        """
        Validate a batch of data records.
        
        Args:
            data_batch: List of data records to validate
            
        Returns:
            List of ValidationResult objects
        """
        return [await self.validate(data) for data in data_batch] 