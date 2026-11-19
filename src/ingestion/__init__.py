"""Ingestion module for configuration-driven dataset loading.
"""

from .loader import (
    load_config,
    load_dataset,
    load_company_data,
    validate_output_template,
)

__all__ = [
    "load_config",
    "load_dataset",
    "load_company_data",
    "validate_output_template",
]
