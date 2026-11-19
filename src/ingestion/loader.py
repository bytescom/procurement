"""Configuration-driven data ingestion module.

Loads tabular datasets for any company according to a YAML configuration file
without company-specific hardcoded logic.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import pandas as pd
import yaml


def get_project_root() -> Path:
    """Locate the project root directory.

    Starts from this file's location and searches parents for project markers.
    Defaults to the current working directory if markers are not found.
    """
    current = Path(__file__).resolve().parent
    for parent in [current] + list(current.parents):
        if (parent / "configs").is_dir() or (parent / "data").is_dir():
            return parent
    return Path.cwd().resolve()


def resolve_path(path: Union[str, Path], project_root: Optional[Union[str, Path]] = None) -> Path:
    """Resolve a relative or absolute file path relative to the project root.

    Args:
        path: Path string or Path object.
        project_root: Optional project root path; defaults to get_project_root().

    Returns:
        Absolute Path object.
    """
    p = Path(path)
    if p.is_absolute():
        return p
    root = Path(project_root) if project_root else get_project_root()
    return (root / p).resolve()


def _validate_config(config: Any, config_path: Union[str, Path]) -> Dict[str, Any]:
    """Validate the shared structure required by the generic ingestion loader."""
    if not isinstance(config, dict):
        raise ValueError(
            f"Configuration at {config_path} must be a dictionary, got {type(config).__name__}"
        )

    company = config.get("company")
    if not isinstance(company, dict):
        raise ValueError(f"Configuration at {config_path} must contain a 'company' mapping.")
    missing_company_fields = [
        field for field in ("id", "name", "reporting_currency")
        if not isinstance(company.get(field), str) or not company[field].strip()
    ]
    if missing_company_fields:
        raise ValueError(
            f"Configuration at {config_path} has missing or empty company fields: "
            f"{missing_company_fields}"
        )

    datasets = config.get("datasets")
    if not isinstance(datasets, dict) or not datasets:
        raise ValueError(f"Configuration at {config_path} must contain a non-empty 'datasets' mapping.")

    for key, spec in datasets.items():
        if not isinstance(spec, dict):
            raise ValueError(f"Dataset specification for '{key}' must be a dictionary.")
        if not isinstance(spec.get("path"), str) or not spec["path"].strip():
            raise ValueError(f"Dataset specification for '{key}' is missing required 'path' field.")

        columns = spec.get("columns")
        if not isinstance(columns, dict) or not columns:
            raise ValueError(f"Dataset specification for '{key}' must contain a non-empty 'columns' mapping.")
        invalid_mappings = [
            semantic for semantic, source in columns.items()
            if not isinstance(semantic, str) or not semantic.strip()
            or not isinstance(source, str) or not source.strip()
        ]
        if invalid_mappings:
            raise ValueError(f"Dataset specification for '{key}' has invalid column mappings: {invalid_mappings}")

        required_columns = spec.get("required_columns")
        if not isinstance(required_columns, list) or any(
            not isinstance(column, str) or not column.strip() for column in required_columns
        ):
            raise ValueError(
                f"Dataset specification for '{key}' must contain 'required_columns' as a list of column names."
            )

        id_column = spec.get("id_column")
        if id_column is not None and (not isinstance(id_column, str) or not id_column.strip()):
            raise ValueError(f"Dataset specification for '{key}' has an invalid 'id_column'.")

    return config


def load_config(config_path: Union[str, Path], project_root: Optional[Union[str, Path]] = None) -> Dict[str, Any]:
    """Load and validate a YAML configuration file.

    Args:
        config_path: Path to the YAML configuration file.
        project_root: Optional project root for resolving relative paths.

    Returns:
        Parsed configuration dictionary.

    Raises:
        FileNotFoundError: If the configuration file does not exist.
        ValueError: If configuration syntax is invalid or required sections are missing.
    """
    resolved_path = resolve_path(config_path, project_root)
    if not resolved_path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {resolved_path}")

    try:
        with open(resolved_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise ValueError(f"Failed to parse YAML configuration at {resolved_path}: {e}") from e

    return _validate_config(config, resolved_path)


def load_dataset(
    file_path: Union[str, Path],
    required_columns: Optional[List[str]] = None,
    dataset_name: Optional[str] = None,
    project_root: Optional[Union[str, Path]] = None,
    **kwargs: Any,
) -> pd.DataFrame:
    """Load a CSV dataset and validate required columns.

    Args:
        file_path: Path to the CSV file.
        required_columns: Optional list of columns that must be present.
        dataset_name: Optional label for logging/error context.
        project_root: Optional project root for relative path resolution.
        **kwargs: Additional parameters passed to pd.read_csv.

    Returns:
        Loaded pandas DataFrame.

    Raises:
        FileNotFoundError: If the CSV file does not exist.
        ValueError: If required columns are missing from the loaded DataFrame.
    """
    resolved_path = resolve_path(file_path, project_root)
    name_str = f" '{dataset_name}'" if dataset_name else ""

    if not resolved_path.is_file():
        raise FileNotFoundError(f"Dataset{name_str} file not found at: {resolved_path}")

    # Default low_memory to False to prevent mixed-type inference warnings
    kwargs.setdefault("low_memory", False)

    try:
        df = pd.read_csv(resolved_path, **kwargs)
    except Exception as e:
        raise RuntimeError(f"Failed to read CSV dataset{name_str} from {resolved_path}: {e}") from e

    if required_columns:
        present_cols = set(df.columns)
        missing_cols = [c for c in required_columns if c not in present_cols]
        if missing_cols:
            raise ValueError(
                f"Dataset{name_str} at {resolved_path} is missing required columns: {missing_cols}. "
                f"Present columns: {list(df.columns)}"
            )

    return df


def load_company_data(
    config_or_path: Union[str, Path, Dict[str, Any]],
    project_root: Optional[Union[str, Path]] = None,
) -> Dict[str, Any]:
    """Ingest all datasets for a company based on a configuration.

    Args:
        config_or_path: Either a configuration dictionary or path to a YAML file.
        project_root: Optional project root for relative path resolution.

    Returns:
        Dictionary containing:
            - 'config': The configuration dictionary.
            - 'company_id': The company ID string.
            - 'company_name': The company name string.
            - 'reporting_currency': The reporting currency.
            - 'datasets': Dictionary of {dataset_key: pd.DataFrame}.

    Raises:
        ValueError: If dataset definitions in config are invalid.
        FileNotFoundError: If any configured dataset file is missing.
    """
    root = Path(project_root) if project_root else get_project_root()

    if isinstance(config_or_path, (str, Path)):
        config = load_config(config_or_path, project_root=root)
    elif isinstance(config_or_path, dict):
        config = _validate_config(config_or_path, "<in-memory configuration>")
    else:
        raise TypeError(f"config_or_path must be a dict or path, got {type(config_or_path).__name__}")

    company_info = config.get("company", {})
    datasets_config = config.get("datasets", {})

    loaded_datasets: Dict[str, pd.DataFrame] = {}

    for key, spec in datasets_config.items():
        if not isinstance(spec, dict):
            raise ValueError(f"Dataset specification for '{key}' must be a dictionary.")

        path = spec.get("path")
        if not path:
            raise ValueError(f"Dataset specification for '{key}' missing required 'path' field.")

        name = spec.get("name", key)
        required_cols = spec.get("required_columns")

        df = load_dataset(
            file_path=path,
            required_columns=required_cols,
            dataset_name=name,
            project_root=root,
        )

        # The semantic column map is part of the company configuration. Validate
        # it against the actual CSV header so misspelled source fields fail early.
        mapped_columns = list(spec["columns"].values())
        id_column = spec.get("id_column")
        if id_column:
            mapped_columns.append(id_column)
        missing_mapped_columns = [column for column in mapped_columns if column not in df.columns]
        if missing_mapped_columns:
            raise ValueError(
                f"Dataset '{name}' at {resolve_path(path, root)} has column mappings that "
                f"reference missing columns: {missing_mapped_columns}. Present columns: {list(df.columns)}"
            )

        loaded_datasets[key] = df

    return {
        "config": config,
        "company_id": company_info.get("id", "UNKNOWN"),
        "company_name": company_info.get("name", "Unknown Company"),
        "reporting_currency": company_info.get("reporting_currency", ""),
        "datasets": loaded_datasets,
    }


def validate_output_template(
    template_path: Optional[Union[str, Path]] = None,
    project_root: Optional[Union[str, Path]] = None,
) -> Dict[str, Any]:
    """Validate the official output template schema.

    Args:
        template_path: Path to output_template.csv. If omitted, checks 'output_template.csv'
                       and 'data/output_template.csv'.
        project_root: Optional project root directory.

    Returns:
        Dictionary with validation status, columns, and sample row count.

    Raises:
        FileNotFoundError: If the output template cannot be found.
        ValueError: If columns or order do not match the official schema.
    """
    root = Path(project_root) if project_root else get_project_root()
    expected_columns = [
        "record_id",
        "include",
        "exclusion_reason",
        "supplier_key",
        "item_key",
        "spend",
        "l1",
        "l2",
        "l3",
        "l4",
        "confidence",
    ]

    resolved_path: Optional[Path] = None
    if template_path:
        p = resolve_path(template_path, root)
        if p.is_file():
            resolved_path = p
    else:
        for candidate in ["output_template.csv", "data/output_template.csv"]:
            p = resolve_path(candidate, root)
            if p.is_file():
                resolved_path = p
                break

    if not resolved_path:
        raise FileNotFoundError("Official output template file not found.")

    df_template = pd.read_csv(resolved_path)
    actual_columns = list(df_template.columns)

    if actual_columns != expected_columns:
        raise ValueError(
            f"Output template schema mismatch.\n"
            f"Expected: {expected_columns}\n"
            f"Actual:   {actual_columns}"
        )

    return {
        "status": "VALID",
        "template_path": str(resolved_path),
        "columns": actual_columns,
        "column_count": len(actual_columns),
        "sample_rows": len(df_template),
    }
