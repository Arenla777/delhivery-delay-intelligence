
from pathlib import Path

import numpy as np
import pandas as pd

from src.shipment_prep import (
    FEATURES,
    PROMISED_HOURS,
    build_features,
    clean_shipments,
)


# ---------------------------------------------------------
# Paths and output folders
# ---------------------------------------------------------

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "outputs"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------
# Helper: write CSV files without an index and round
# decimal values to four places
# ---------------------------------------------------------

def write_csv(df, path):
    output = df.copy()

    float_columns = output.select_dtypes(
        include=["float", "float32", "float64"]
    ).columns

    output[float_columns] = output[float_columns].round(4)

    output.to_csv(path, index=False)


# ---------------------------------------------------------
# 1. Load the source data
# ---------------------------------------------------------

raw = pd.read_csv(DATA_DIR / "raw_shipments.csv")
hubs = pd.read_csv(DATA_DIR / "hubs.csv")

raw_rows = len(raw)

# Remove exact duplicates for the "before cleaning" checks.
raw_dedup = raw.drop_duplicates().copy()

duplicate_rows_removed = raw_rows - len(raw_dedup)
rows_after_dedup = len(raw_dedup)


# ---------------------------------------------------------
# 2. Capture quality metrics before imputation
# ---------------------------------------------------------

missing_distance_km_before = int(
    raw_dedup["distance_km"].isna().sum()
)

# Sentinel weights are not counted as missing here.
missing_parcel_weight_before = int(
    raw_dedup["parcel_weight_kg"].isna().sum()
)

sentinel_weight_rows = int(
    (
        pd.to_numeric(
            raw_dedup["parcel_weight_kg"],
            errors="coerce",
        ) > 100
    ).sum()
)

# The generator writes declared values as text, such as ₹1,290.
# Count missing entries before cleaning.
missing_declared_value_before = int(
    raw_dedup["declared_value"].isna().sum()
)

missing_payment_mode_before = int(
    raw_dedup["payment_mode"].isna().sum()
)

service_type_spellings_before = int(
    raw_dedup["service_type"].nunique(dropna=True)
)


# ---------------------------------------------------------
# 3. Clean shipments and calculate medians
# ---------------------------------------------------------

clean, medians = clean_shipments(raw)

service_type_spellings_after = int(
    clean["service_type"].nunique(dropna=True)
)


# ---------------------------------------------------------
# 4. Build features and attach hub information
# ---------------------------------------------------------

modelling_table = build_features(clean, hubs)

rows_after_hub_merge = len(modelling_table)

# Add the target only after the feature engineering step.
modelling_table["delivery_hours"] = pd.to_numeric(
    modelling_table["delivery_hours"],
    errors="coerce",
)

modelling_table["is_delayed"] = (
    modelling_table["delivery_hours"]
    > modelling_table["promised_hours"]
).astype("int64")


# ---------------------------------------------------------
# 5. Calculate delay and delivery-hours metrics
# ---------------------------------------------------------

delayed_count = int(modelling_table["is_delayed"].sum())

delay_rate = float(modelling_table["is_delayed"].mean())

hours = modelling_table["delivery_hours"].dropna()

hours_q1 = float(hours.quantile(0.25))
hours_q3 = float(hours.quantile(0.75))

hours_iqr = hours_q3 - hours_q1

hours_lower_bound = hours_q1 - 1.5 * hours_iqr
hours_upper_bound = hours_q3 + 1.5 * hours_iqr

hours_outlier_count = int(
    (
        (modelling_table["delivery_hours"] < hours_lower_bound)
        | (modelling_table["delivery_hours"] > hours_upper_bound)
    ).sum()
)


# ---------------------------------------------------------
# 6. Declared-value statistics on the median-filled column
# ---------------------------------------------------------

declared_values = modelling_table["declared_value_inr"]
log_declared_values = np.log(declared_values)

declared_value_mean_inr = float(declared_values.mean())
declared_value_median_inr = float(declared_values.median())

log_declared_value_mean = float(log_declared_values.mean())
log_declared_value_median = float(log_declared_values.median())


# ---------------------------------------------------------
# 7. Check missing values in model features and target
# ---------------------------------------------------------

columns_to_check = FEATURES + ["delivery_hours", "is_delayed"]

nulls_in_features_and_target = int(
    modelling_table[columns_to_check].isna().sum().sum()
)


# ---------------------------------------------------------
# 8. Save the complete modelling table
# ---------------------------------------------------------

write_csv(
    modelling_table,
    DATA_DIR / "modelling_table.csv",
)


# ---------------------------------------------------------
# 9. Write Part A checks
# ---------------------------------------------------------

checks = [
    ("raw_rows", raw_rows),
    ("duplicate_rows_removed", duplicate_rows_removed),
    ("rows_after_dedup", rows_after_dedup),
    ("missing_distance_km_before", missing_distance_km_before),
    ("missing_parcel_weight_before", missing_parcel_weight_before),
    ("sentinel_weight_rows", sentinel_weight_rows),
    ("missing_declared_value_before", missing_declared_value_before),
    ("missing_payment_mode_before", missing_payment_mode_before),
    ("service_type_spellings_before", service_type_spellings_before),
    ("service_type_spellings_after", service_type_spellings_after),
    ("median_distance_km", medians["distance_km"]),
    ("median_parcel_weight_kg", medians["parcel_weight_kg"]),
    ("median_declared_value_inr", medians["declared_value_inr"]),
    ("rows_after_hub_merge", rows_after_hub_merge),
    ("nulls_in_features_and_target", nulls_in_features_and_target),
    ("delayed_count", delayed_count),
    ("delay_rate", delay_rate),
    ("hours_q1", hours_q1),
    ("hours_q3", hours_q3),
    ("hours_lower_bound", hours_lower_bound),
    ("hours_upper_bound", hours_upper_bound),
    ("hours_outlier_count", hours_outlier_count),
    ("declared_value_mean_inr", declared_value_mean_inr),
    ("declared_value_median_inr", declared_value_median_inr),
    ("log_declared_value_mean", log_declared_value_mean),
    ("log_declared_value_median", log_declared_value_median),
]

checks_df = pd.DataFrame(checks, columns=["metric", "value"])

write_csv(
    checks_df,
    OUTPUT_DIR / "part_a_checks.csv",
)


# ---------------------------------------------------------
# 10. Delay evidence by service type, hub tier and distance
# ---------------------------------------------------------

# Distance bands use right-inclusive intervals:
# (0, 200], (200, 400], (400, 600], (600, 2000].
modelling_table["distance_band"] = pd.cut(
    modelling_table["distance_km"],
    bins=[0, 200, 400, 600, 2000],
    labels=["0-200", "201-400", "401-600", "601+"],
    right=True,
    include_lowest=True,
)

group_rows = []

grouping_specs = [
    ("service_type", "service_type"),
    ("hub_tier", "hub_tier"),
    ("distance_band", "distance_band"),
]

for dimension, column in grouping_specs:
    grouped = modelling_table.groupby(
        column,
        observed=True,
        dropna=False,
        sort=True,
    )

    for group_name, group_df in grouped:
        shipments = len(group_df)
        delayed = int(group_df["is_delayed"].sum())

        group_rows.append(
            {
                "dimension": dimension,
                "group": group_name,
                "shipments": shipments,
                "delayed": delayed,
                "delay_rate": delayed / shipments if shipments else 0.0,
                "mean_hours": group_df["delivery_hours"].mean(),
            }
        )

delay_by_group = pd.DataFrame(
    group_rows,
    columns=[
        "dimension",
        "group",
        "shipments",
        "delayed",
        "delay_rate",
        "mean_hours",
    ],
)

# Use explicit business ordering for consistent output.
dimension_order = {
    "service_type": 0,
    "hub_tier": 1,
    "distance_band": 2,
}

service_order = {
    "Economy": 0,
    "Express": 1,
    "Standard": 2,
}

distance_order = {
    "0-200": 0,
    "201-400": 1,
    "401-600": 2,
    "601+": 3,
}


def group_sort_key(row):
    dimension = row["dimension"]
    group = row["group"]

    if dimension == "service_type":
        group_order = service_order.get(str(group), 99)
    elif dimension == "hub_tier":
        try:
            group_order = int(group)
        except (TypeError, ValueError):
            group_order = 99
    else:
        group_order = distance_order.get(str(group), 99)

    return dimension_order[dimension], group_order


delay_by_group["_sort_key"] = delay_by_group.apply(
    group_sort_key,
    axis=1,
)

delay_by_group = (
    delay_by_group
    .sort_values("_sort_key")
    .drop(columns="_sort_key")
    .reset_index(drop=True)
)

write_csv(
    delay_by_group,
    OUTPUT_DIR / "part_a_delay_by_group.csv",
)


# ---------------------------------------------------------
# 11. Correlation matrix of numeric modelling-table columns
# ---------------------------------------------------------

numeric_table = modelling_table.select_dtypes(include="number")

correlations = numeric_table.corr()

# Keep the feature names as the first CSV column.
correlations.index.name = "feature"
correlations = correlations.reset_index()

write_csv(
    correlations,
    OUTPUT_DIR / "part_a_correlations.csv",
)


# ---------------------------------------------------------
# 12. Print a concise validation summary
# ---------------------------------------------------------

print("PART A COMPLETE")
print("-" * 45)
print(f"Raw rows: {raw_rows}")
print(f"Duplicate rows removed: {duplicate_rows_removed}")
print(f"Rows after cleaning: {len(clean)}")
print(f"Rows after hub merge: {rows_after_hub_merge}")
print(f"Delayed shipments: {delayed_count}")
print(f"Delay rate: {delay_rate:.4f}")
print(f"Missing feature/target values: {nulls_in_features_and_target}")
print(f"Feature columns: {len(FEATURES)}")
print(f"Medians: {medians}")
print()
print("Files created:")
print("  data/modelling_table.csv")
print("  outputs/part_a_checks.csv")
print("  outputs/part_a_delay_by_group.csv")
print("  outputs/part_a_correlations.csv")
