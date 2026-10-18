"""
core/data_validation.py
Data Validation, Column Mapping, and Data Profiling Engine for StockSense AI.
Inspects CSV files, verifies date continuity, identifies duplicates,
handles missing fields transparently, and applies zero-trust input defense.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Tuple
import pandas as pd
import numpy as np

from tools.policy_retriever import PolicyRetriever


@dataclass
class ColumnMapping:
    date_col: Optional[str] = None
    product_id_col: Optional[str] = None
    store_id_col: Optional[str] = None
    sales_col: Optional[str] = None
    inventory_col: Optional[str] = None
    price_col: Optional[str] = None
    incoming_col: Optional[str] = None
    category_col: Optional[str] = None


@dataclass
class DataProfile:
    total_rows: int = 0
    total_columns: int = 0
    columns: List[str] = field(default_factory=list)
    missing_by_col: Dict[str, int] = field(default_factory=dict)
    duplicate_rows_count: int = 0
    date_min: Optional[str] = None
    date_max: Optional[str] = None
    date_coverage_days: int = 0
    unique_products_count: int = 0
    unique_stores_count: int = 0
    sample_products: List[str] = field(default_factory=list)
    sample_stores: List[str] = field(default_factory=list)
    has_negative_sales: bool = False
    has_negative_inventory: bool = False
    flagged_adversarial_rows: int = 0
    validation_status: str = "PENDING"  # VALID, WARNINGS, INVALID
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)


CANDIDATE_PATTERNS = {
    "date_col": ["date", "datetime", "timestamp", "day", "record_date"],
    "product_id_col": ["product_id", "product id", "sku", "item_id", "item", "product"],
    "store_id_col": ["store_id", "store id", "store", "shop_id", "location_id", "branch"],
    "sales_col": ["units_sold", "units sold", "sales", "qty_sold", "quantity_sold", "demand", "units_sold (sales)"],
    "inventory_col": ["inventory_level", "inventory level", "stock", "current_stock", "on_hand", "inventory"],
    "price_col": ["price", "unit_price", "selling_price", "unit price"],
    "incoming_col": ["units_ordered", "units ordered", "incoming_stock", "in_transit", "on_order"],
    "category_col": ["category", "product_category", "dept", "department"],
}


def auto_detect_columns(columns: Any) -> ColumnMapping:
    """
    Intelligently maps uploaded CSV column headers to canonical internal identifiers.
    Matches case-insensitively and handles spaces/underscores.
    Supports list of strings, DataFrame, or Index.
    """
    if hasattr(columns, "columns"):
        col_list = list(columns.columns)
    else:
        col_list = list(columns)
    mapping = ColumnMapping()
    normalized = {str(c).strip().lower().replace("-", "_"): str(c) for c in col_list}

    for target_field, candidates in CANDIDATE_PATTERNS.items():
        matched = None
        for cand in candidates:
            cand_norm = cand.lower().replace("-", "_")
            if cand_norm in normalized:
                matched = normalized[cand_norm]
                break
        if not matched:
            # Fuzzy match: check substring if not matched exactly
            for norm_key, orig_col in normalized.items():
                for cand in candidates:
                    if cand.lower() in norm_key:
                        matched = orig_col
                        break
                if matched:
                    break

        setattr(mapping, target_field, matched)

    return mapping


def profile_and_validate_dataset(
    df: pd.DataFrame,
    mapping: ColumnMapping
) -> Tuple[DataProfile, pd.DataFrame]:
    """
    Validates CSV schema, date continuity, duplicates, missing cells, and adversarial injections.
    Returns a clean DataProfile and a sanitized/prepared DataFrame.
    """
    profile = DataProfile()
    profile.total_rows = len(df)
    profile.total_columns = len(df.columns)
    profile.columns = list(df.columns)

    if df.empty:
        profile.validation_status = "INVALID"
        profile.errors.append("Uploaded dataset is empty (0 rows).")
        return profile, df

    # Check missing values
    profile.missing_by_col = {col: int(df[col].isna().sum()) for col in df.columns}

    # Verify essential columns
    if not mapping.date_col or mapping.date_col not in df.columns:
        profile.errors.append("Date column is missing or unmapped.")
    if not mapping.product_id_col or mapping.product_id_col not in df.columns:
        profile.errors.append("Product ID column is missing or unmapped.")
    if not mapping.sales_col or mapping.sales_col not in df.columns:
        profile.errors.append("Sales/Demand column is missing or unmapped.")

    if profile.errors:
        profile.validation_status = "INVALID"
        return profile, df

    clean_df = df.copy()

    # Zero-trust inspection of string fields for prompt injection
    flagged_rows = 0
    string_cols = clean_df.select_dtypes(include=["object", "string"]).columns
    for col in string_cols:
        for idx, val in clean_df[col].dropna().items():
            if isinstance(val, str):
                _, is_flagged = PolicyRetriever.sanitize_untrusted_input(val)
                if is_flagged:
                    flagged_rows += 1
    profile.flagged_adversarial_rows = flagged_rows
    if flagged_rows > 0:
        profile.warnings.append(
            f"Neutralized {flagged_rows} cell(s) containing adversarial prompt injection patterns."
        )

    # Date parsing and validation
    try:
        clean_df["_parsed_date"] = pd.to_datetime(clean_df[mapping.date_col], errors="coerce")
        invalid_dates = clean_df["_parsed_date"].isna().sum()
        if invalid_dates > 0:
            profile.warnings.append(f"Found {invalid_dates} rows with unparseable or malformed dates.")
            clean_df = clean_df.dropna(subset=["_parsed_date"]).copy()

        if not clean_df.empty:
            min_d = clean_df["_parsed_date"].min()
            max_d = clean_df["_parsed_date"].max()
            profile.date_min = min_d.strftime("%Y-%m-%d")
            profile.date_max = max_d.strftime("%Y-%m-%d")
            profile.date_coverage_days = (max_d - min_d).days + 1
    except Exception as e:
        profile.errors.append(f"Date conversion error: {str(e)}")
        profile.validation_status = "INVALID"
        return profile, df

    # Sales column validation
    clean_df["_parsed_sales"] = pd.to_numeric(clean_df[mapping.sales_col], errors="coerce")
    missing_sales = clean_df["_parsed_sales"].isna().sum()
    if missing_sales > 0:
        profile.warnings.append(
            f"Found {missing_sales} rows with missing sales. Records excluded from time-series aggregation."
        )
        clean_df = clean_df.dropna(subset=["_parsed_sales"]).copy()

    negative_sales_cnt = (clean_df["_parsed_sales"] < 0).sum()
    if negative_sales_cnt > 0:
        profile.has_negative_sales = True
        profile.warnings.append(
            f"Detected {negative_sales_cnt} negative sales values. Clipped to zero per retail audit standards."
        )
        clean_df["_parsed_sales"] = clean_df["_parsed_sales"].clip(lower=0.0)

    # Inventory column validation (if present)
    if mapping.inventory_col and mapping.inventory_col in clean_df.columns:
        clean_df["_parsed_inventory"] = pd.to_numeric(clean_df[mapping.inventory_col], errors="coerce")
        neg_inv = (clean_df["_parsed_inventory"] < 0).sum()
        if neg_inv > 0:
            profile.has_negative_inventory = True
            profile.warnings.append(
                f"Detected {neg_inv} negative inventory values. Handled as data quality discrepancies."
            )
    else:
        profile.warnings.append(
            "Current stock column is absent from uploaded data. User must configure assumptions or provide inventory."
        )

    # Optional price, incoming stock, and category parsing
    if mapping.price_col and mapping.price_col in clean_df.columns:
        clean_df["_parsed_price"] = pd.to_numeric(clean_df[mapping.price_col], errors="coerce")
    if mapping.incoming_col and mapping.incoming_col in clean_df.columns:
        clean_df["_parsed_incoming"] = pd.to_numeric(clean_df[mapping.incoming_col], errors="coerce")
    if mapping.category_col and mapping.category_col in clean_df.columns:
        clean_df["_parsed_category"] = clean_df[mapping.category_col].astype(str)

    # Product and Store cardinality
    clean_df["_product_id"] = clean_df[mapping.product_id_col].astype(str)
    unique_prods = clean_df["_product_id"].unique().tolist()
    profile.unique_products_count = len(unique_prods)
    profile.sample_products = unique_prods[:5]

    if mapping.store_id_col and mapping.store_id_col in clean_df.columns:
        clean_df["_store_id"] = clean_df[mapping.store_id_col].astype(str)
        unique_stores = clean_df["_store_id"].unique().tolist()
        profile.unique_stores_count = len(unique_stores)
        profile.sample_stores = unique_stores[:5]
    else:
        clean_df["_store_id"] = "ALL_STORES"
        profile.unique_stores_count = 1
        profile.sample_stores = ["ALL_STORES"]

    # Check duplicates for (Date, Product, Store)
    dup_mask = clean_df.duplicated(subset=["_parsed_date", "_product_id", "_store_id"], keep=False)
    dup_count = int(dup_mask.sum())
    profile.duplicate_rows_count = dup_count
    if dup_count > 0:
        profile.warnings.append(
            f"Detected {dup_count} duplicate rows for (Date, Product, Store). Aggregating sales by sum."
        )
        # Aggregate duplicates safely
        agg_cols = {"_parsed_sales": "sum"}
        if "_parsed_inventory" in clean_df.columns:
            agg_cols["_parsed_inventory"] = "last"
        if "_parsed_price" in clean_df.columns:
            agg_cols["_parsed_price"] = "last"
        if "_parsed_incoming" in clean_df.columns:
            agg_cols["_parsed_incoming"] = "last"
        if "_parsed_category" in clean_df.columns:
            agg_cols["_parsed_category"] = "last"

        clean_df = clean_df.groupby(
            ["_parsed_date", "_product_id", "_store_id"], as_index=False
        ).agg(agg_cols)

    # Final status determination
    if profile.errors:
        profile.validation_status = "INVALID"
    elif profile.warnings:
        profile.validation_status = "WARNINGS"
    else:
        profile.validation_status = "VALID"

    return profile, clean_df
