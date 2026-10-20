"""StatBot Pro - Advanced Multi-Format Dataset Engine.

Supports CSV, TSV, Excel (multi-sheet), JSON, JSONL, Parquet, and Arrow/Feather.
Provides smart type inference, robust encoding fallbacks, data profiling,
cleaning operations, and multi-format exports.
"""
from __future__ import annotations

import csv
import io
import os
import re
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd


ENCODING_CANDIDATES = ["utf-8", "utf-8-sig", "latin-1", "cp1252", "iso-8859-1"]


def detect_delimiter(sample_text: str) -> str:
    """Detect delimiter using csv.Sniffer with safe heuristic fallback."""
    try:
        sniffer = csv.Sniffer()
        dialect = sniffer.sniff(sample_text[:4096])
        if dialect.delimiter in [",", ";", "\t", "|"]:
            return dialect.delimiter
    except Exception:
        pass

    # Heuristic counts in first few lines
    lines = [line for line in sample_text.splitlines() if line.strip()][:5]
    if not lines:
        return ","
    header = lines[0]
    counts = {
        ",": header.count(","),
        ";": header.count(";"),
        "\t": header.count("\t"),
        "|": header.count("|"),
    }
    best = max(counts, key=counts.get)  # type: ignore
    return best if counts[best] > 0 else ","


def get_excel_sheet_names(source: Union[str, bytes, io.BytesIO]) -> List[str]:
    """Retrieve sheet names from an Excel file without loading entire data."""
    try:
        if isinstance(source, (bytes, bytearray)):
            source = io.BytesIO(source)
        excel_file = pd.ExcelFile(source)
        return excel_file.sheet_names
    except Exception:
        return ["Sheet1"]


def clean_currency_or_number_series(series: pd.Series) -> pd.Series:
    """Attempt to parse currency strings, percentages, and formatted numbers to float."""
    if not pd.api.types.is_object_dtype(series) and not pd.api.types.is_string_dtype(series):
        return series

    # Quick sample check if values look like numbers with currency/commas
    non_null = series.dropna().astype(str).str.strip()
    if non_null.empty:
        return series

    sample = non_null.iloc[: min(50, len(non_null))]
    # Regex looking for e.g. $1,234.50 or 149,934.18 INR or 12.5%
    pattern = re.compile(r"^[\$€£₹¥]?\s*[-+]?[0-9]{1,3}(?:,[0-9]{3})*(?:\.[0-9]+)?\s*[\%a-zA-Z]*$")
    matches = sum(bool(pattern.match(val)) for val in sample)
    if matches / len(sample) >= 0.6:
        # Strip currency symbols, spaces, commas, and percentage signs
        cleaned = (
            non_null.str.replace(r"[\$€£₹¥,]", "", regex=True)
            .str.replace(r"\s+[A-Za-z]+$", "", regex=True)
            .str.replace(r"\%$", "", regex=True)
            .str.strip()
        )
        converted = pd.to_numeric(cleaned, errors="coerce")
        if converted.notna().sum() >= max(1, len(series) // 2):
            result = series.copy()
            result.loc[converted.index] = converted
            return pd.to_numeric(result, errors="coerce")
    return series


def sanitize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Clean column names, normalize text, parse dates & numericals safely."""
    # Deduplicate and strip column names
    clean_cols: List[str] = []
    seen: Dict[str, int] = {}
    for col in df.columns:
        norm = re.sub(r"\s+", " ", str(col).strip())
        if not norm:
            norm = "unnamed"
        if norm in seen:
            seen[norm] += 1
            norm = f"{norm}_{seen[norm]}"
        else:
            seen[norm] = 0
        clean_cols.append(norm)
    df.columns = clean_cols

    # Strip text whitespace
    for col in df.select_dtypes(include=["object", "string"]).columns:
        df[col] = df[col].astype("string").str.strip()

    # Try numeric conversion
    for col in df.columns:
        # If already numeric or datetime, leave as is
        if pd.api.types.is_numeric_dtype(df[col]) or pd.api.types.is_datetime64_any_dtype(df[col]):
            continue

        # Try currency / formatted number
        df[col] = clean_currency_or_number_series(df[col])

        # Try standard numeric conversion
        converted = pd.to_numeric(df[col], errors="coerce")
        if converted.notna().sum() >= max(2, int(len(df) * 0.5)):
            df[col] = converted
            continue

        # Try datetime conversion if column name contains date/time/year or values match
        col_lower = str(col).lower()
        if any(term in col_lower for term in ["date", "time", "timestamp", "year", "month", "day"]):
            try:
                dt_converted = pd.to_datetime(df[col], errors="coerce")
                if dt_converted.notna().sum() >= max(2, int(len(df) * 0.5)):
                    df[col] = dt_converted
            except Exception:
                pass

    return df


def load_dataset(
    source: Union[str, bytes, io.BytesIO, Any],
    filename: Optional[str] = None,
    sheet_name: Optional[Union[str, int]] = 0,
    encoding: Optional[str] = None,
    delimiter: Optional[str] = None,
) -> pd.DataFrame:
    """Universal dataset reader supporting CSV, TSV, Excel, JSON, Parquet, and Arrow."""
    # Determine effective filename
    if filename is None or filename == "dataset.csv":
        if isinstance(source, str):
            filename = os.path.basename(source)
        elif hasattr(source, "name") and getattr(source, "name"):
            filename = getattr(source, "name")
        else:
            filename = filename or "dataset.csv"

    ext = os.path.splitext(filename)[1].lower()

    # Helper to get raw bytes
    if isinstance(source, (bytes, bytearray)):
        data_bytes = bytes(source)
    elif isinstance(source, io.BytesIO):
        data_bytes = source.getvalue()
    elif isinstance(source, str) and os.path.exists(source):
        with open(source, "rb") as f:
            data_bytes = f.read()
    else:
        raise ValueError(f"Invalid dataset source: {source}")

    # 1. Excel files (.xlsx, .xls, .xlsm)
    if ext in [".xlsx", ".xls", ".xlsm"]:
        bio = io.BytesIO(data_bytes)
        sheet = 0 if sheet_name is None else sheet_name
        df = pd.read_excel(bio, sheet_name=sheet)
        return sanitize_dataframe(df)

    # 2. Parquet files (.parquet, .pq)
    if ext in [".parquet", ".pq"]:
        bio = io.BytesIO(data_bytes)
        df = pd.read_parquet(bio)
        return sanitize_dataframe(df)

    # 3. Arrow / Feather files (.arrow, .feather)
    if ext in [".feather", ".arrow"]:
        bio = io.BytesIO(data_bytes)
        df = pd.read_feather(bio)
        return sanitize_dataframe(df)

    # 4. JSON / JSONL files (.json, .jsonl, .ndjson)
    if ext in [".json", ".jsonl", ".ndjson"]:
        bio = io.BytesIO(data_bytes)
        try:
            # Try records orientation or standard json
            df = pd.read_json(bio)
        except Exception:
            bio.seek(0)
            try:
                # Try lines orientation (JSONL)
                df = pd.read_json(bio, lines=True)
            except Exception:
                bio.seek(0)
                import json
                raw_json = json.loads(bio.read().decode("utf-8", errors="ignore"))
                if isinstance(raw_json, list):
                    df = pd.json_normalize(raw_json)
                elif isinstance(raw_json, dict):
                    # Check if keys contain lists
                    first_val = next(iter(raw_json.values()), None)
                    if isinstance(first_val, list):
                        df = pd.DataFrame(raw_json)
                    else:
                        df = pd.json_normalize(raw_json)
                else:
                    raise ValueError("Unrecognized JSON structure.")
        return sanitize_dataframe(df)

    # 5. Delimited Text (CSV, TSV, TXT, etc.)
    encodings_to_try = [encoding] if encoding else ENCODING_CANDIDATES
    last_err: Optional[Exception] = None

    for enc in encodings_to_try:
        try:
            text = data_bytes.decode(enc)
            delim = delimiter or detect_delimiter(text)
            bio = io.StringIO(text)
            df = pd.read_csv(bio, sep=delim, on_bad_lines="skip")
            return sanitize_dataframe(df)
        except Exception as e:
            last_err = e
            continue

    raise ValueError(f"Could not parse file '{filename}'. Last error: {last_err}")


def get_dataset_summary(df: pd.DataFrame) -> Dict[str, Any]:
    """Compute rich dataset metrics and health indicator."""
    total_cells = max(len(df) * max(len(df.columns), 1), 1)
    missing_cells = int(df.isna().sum().sum())
    duplicate_rows = int(df.duplicated().sum())

    # Data quality score: penalizes missing cells and duplicate rows
    missing_ratio = missing_cells / total_cells
    dup_ratio = duplicate_rows / max(len(df), 1)
    quality_score = max(0, min(100, int(round((1.0 - (missing_ratio * 0.7 + dup_ratio * 0.3)) * 100))))

    # Memory usage formatting
    mem_bytes = df.memory_usage(deep=True).sum()
    if mem_bytes >= 1024 * 1024:
        mem_str = f"{mem_bytes / (1024 * 1024):.1f} MB"
    else:
        mem_str = f"{mem_bytes / 1024:.1f} KB"

    # Column categorization
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    datetime_cols = df.select_dtypes(include=["datetime", "datetimetz"]).columns.tolist()
    bool_cols = df.select_dtypes(include="bool").columns.tolist()
    categorical_cols = [c for c in df.columns if c not in numeric_cols and c not in datetime_cols and c not in bool_cols]

    return {
        "rows": len(df),
        "columns": len(df.columns),
        "missing_cells": missing_cells,
        "missing_percentage": round((missing_cells / total_cells) * 100, 2),
        "duplicate_rows": duplicate_rows,
        "duplicate_percentage": round((duplicate_rows / max(len(df), 1)) * 100, 2),
        "quality_score": quality_score,
        "memory_usage": mem_str,
        "numeric_count": len(numeric_cols),
        "categorical_count": len(categorical_cols),
        "datetime_count": len(datetime_cols),
        "numeric_cols": numeric_cols,
        "categorical_cols": categorical_cols,
        "datetime_cols": datetime_cols,
        "bool_cols": bool_cols,
    }


def get_column_profiles(df: pd.DataFrame) -> pd.DataFrame:
    """Generate detailed column-by-column diagnostic health profile."""
    records = []
    n = len(df)
    for col in df.columns:
        series = df[col]
        missing = int(series.isna().sum())
        fill_pct = f"{((n - missing) / max(n, 1)) * 100:.1f}%"
        nunique = int(series.nunique(dropna=True))

        dtype_str = str(series.dtype)
        if pd.api.types.is_numeric_dtype(series):
            cat_type = "Numeric"
            non_null = series.dropna()
            stat_summary = f"μ: {non_null.mean():.2f} | σ: {non_null.std():.2f}" if not non_null.empty else "N/A"
        elif pd.api.types.is_datetime64_any_dtype(series):
            cat_type = "Datetime"
            non_null = series.dropna()
            stat_summary = f"{non_null.min().strftime('%Y-%m-%d')} → {non_null.max().strftime('%Y-%m-%d')}" if not non_null.empty else "N/A"
        elif pd.api.types.is_bool_dtype(series):
            cat_type = "Boolean"
            top_val = series.mode().iloc[0] if not series.empty else "N/A"
            stat_summary = f"Mode: {top_val}"
        else:
            cat_type = "Categorical"
            top_val = series.mode().iloc[0] if not series.dropna().empty else "N/A"
            stat_summary = f"Top: {str(top_val)[:20]}"

        records.append({
            "Column Name": col,
            "Classification": cat_type,
            "Data Type": dtype_str,
            "Filled %": fill_pct,
            "Missing": missing,
            "Unique Count": nunique,
            "Key Metric / Range": stat_summary,
        })
    return pd.DataFrame(records)


def apply_transformations(
    df: pd.DataFrame,
    drop_columns: Optional[List[str]] = None,
    drop_duplicates: bool = False,
    impute_strategy: str = "none",  # none, drop, mean_median, mode
    custom_filter_col: Optional[str] = None,
    custom_filter_val: Optional[Any] = None,
) -> pd.DataFrame:
    """Perform non-destructive transformation operations on a dataframe copy."""
    transformed = df.copy()

    # Drop selected columns
    if drop_columns:
        transformed = transformed.drop(columns=[c for c in drop_columns if c in transformed.columns])

    # Drop duplicates
    if drop_duplicates:
        transformed = transformed.drop_duplicates()

    # Impute missing values
    if impute_strategy == "drop":
        transformed = transformed.dropna()
    elif impute_strategy == "mean_median":
        for col in transformed.select_dtypes(include="number").columns:
            median_val = transformed[col].median()
            transformed[col] = transformed[col].fillna(median_val)
        for col in transformed.select_dtypes(include=["object", "string", "category"]).columns:
            mode_series = transformed[col].mode()
            if not mode_series.empty:
                transformed[col] = transformed[col].fillna(mode_series.iloc[0])
    elif impute_strategy == "mode":
        for col in transformed.columns:
            mode_series = transformed[col].mode()
            if not mode_series.empty:
                transformed[col] = transformed[col].fillna(mode_series.iloc[0])

    # Filter
    if custom_filter_col and custom_filter_col in transformed.columns and custom_filter_val is not None:
        if isinstance(custom_filter_val, list):
            transformed = transformed[transformed[custom_filter_col].isin(custom_filter_val)]
        elif isinstance(custom_filter_val, tuple) and len(custom_filter_val) == 2:
            # Range filter
            low, high = custom_filter_val
            transformed = transformed[(transformed[custom_filter_col] >= low) & (transformed[custom_filter_col] <= high)]
        else:
            transformed = transformed[transformed[custom_filter_col] == custom_filter_val]

    return transformed


def export_dataframe(df: pd.DataFrame, file_format: str = "csv") -> Tuple[bytes, str, str]:
    """Export dataframe to bytes with appropriate filename extension and MIME type."""
    fmt = file_format.lower()
    if fmt == "excel" or fmt == "xlsx":
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine="openpyxl") as writer:
            df.to_excel(writer, index=False, sheet_name="Data")
        return output.getvalue(), "exported_dataset.xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    elif fmt == "json":
        json_str = df.to_json(orient="records", indent=2, date_format="iso")
        return json_str.encode("utf-8"), "exported_dataset.json", "application/json"
    elif fmt == "parquet":
        output = io.BytesIO()
        df.to_parquet(output, index=False)
        return output.getvalue(), "exported_dataset.parquet", "application/octet-stream"
    else:
        # Default CSV
        csv_str = df.to_csv(index=False)
        return csv_str.encode("utf-8"), "exported_dataset.csv", "text/csv"
