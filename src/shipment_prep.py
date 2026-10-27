
import numpy as np
import pandas as pd


PROMISED_HOURS = {
    "Express": 36,
    "Standard": 60,
    "Economy": 96,
}

FEATURES = [
    "promised_hours",
    "distance_km",
    "parcel_weight_kg",
    "log_declared_value",
    "n_stops",
    "rain_flag",
    "sale_event_flag",
    "is_cod",
    "is_weekend",
    "dispatch_hour",
    "hub_tier",
    "hub_daily_capacity",
    "label_print_batch",
    "scanner_build_no",
    "tape_roll_no",
    "sticker_serial_mod",
    "shelf_slot_no",
]

NOISE_COLUMNS = [
    "label_print_batch",
    "scanner_build_no",
    "tape_roll_no",
    "sticker_serial_mod",
    "shelf_slot_no",
]


def clean_shipments(raw, medians=None):
    """
    Clean shipment records and return the cleaning medians.

    Parameters
    ----------
    raw : pandas.DataFrame
        Raw shipment records.
    medians : dict or None
        Previously calculated medians. Supply these when cleaning
        new shipments so that the training medians are reused.

    Returns
    -------
    clean : pandas.DataFrame
        Cleaned shipment records.
    medians : dict
        Medians for distance_km, parcel_weight_kg,
        and declared_value_inr.
    """
    clean = raw.copy()

    # Remove exact duplicate rows.
    clean = clean.drop_duplicates().copy()

    # Standardize service type.
    clean["service_type"] = (
        clean["service_type"]
        .astype("string")
        .str.strip()
        .str.title()
    )

    # Replace missing payment modes.
    clean["payment_mode"] = clean["payment_mode"].fillna("Unknown")

    # Convert declared value to numeric.
    # The source CSV may contain either the actual rupee symbol
    # or the literal text '\u20b9'.
    value_text = (
        clean["declared_value"]
        .astype("string")
        .str.replace(r"\\u20b9", "", regex=True)
        .str.replace("₹", "", regex=False)
        .str.replace(",", "", regex=False)
        .str.strip()
    )

    clean["declared_value_inr"] = pd.to_numeric(
        value_text,
        errors="coerce",
    ).astype("float64")

    clean = clean.drop(columns=["declared_value"])

    # Convert dispatch time to datetime.
    clean["dispatch_time"] = pd.to_datetime(
        clean["dispatch_time"],
        errors="coerce",
    )

    # Treat implausibly high parcel weights as scanner faults.
    clean["parcel_weight_kg"] = pd.to_numeric(
        clean["parcel_weight_kg"],
        errors="coerce",
    ).astype("float64")

    clean.loc[
        clean["parcel_weight_kg"] > 100,
        "parcel_weight_kg",
    ] = np.nan

    median_columns = [
        "distance_km",
        "parcel_weight_kg",
        "declared_value_inr",
    ]

    # Ensure all columns use numeric floating-point types.
    for column in median_columns:
        clean[column] = pd.to_numeric(
            clean[column],
            errors="coerce",
        ).astype("float64")

        clean[column] = clean[column].replace(
            [np.inf, -np.inf],
            np.nan,
        )

    # Calculate medians from the cleaned training data, or reuse
    # the supplied training medians when processing new shipments.
    if medians is None:
        medians = {
            column: float(clean[column].median())
            for column in median_columns
        }
    else:
        medians = {
            column: float(medians[column])
            for column in median_columns
        }

    # Impute missing numeric values.
    for column in median_columns:
        clean[column] = clean[column].fillna(medians[column])

    return clean, medians


def build_features(clean, hubs):
    """
    Merge hub attributes and engineer the required model features.

    Parameters
    ----------
    clean : pandas.DataFrame
        Cleaned shipment records returned by clean_shipments().
    hubs : pandas.DataFrame
        Hub reference data.

    Returns
    -------
    pandas.DataFrame
        Shipment records with engineered features.
    """
    result = clean.copy()

    hub_columns = [
        "hub_id",
        "hub_city",
        "region",
        "hub_tier",
        "hub_daily_capacity",
    ]

    # Merge hub attributes without dropping unmatched shipments.
    result = result.merge(
        hubs[hub_columns],
        on="hub_id",
        how="left",
        validate="many_to_one",
    )

    # Promised delivery time based on service type.
    result["promised_hours"] = (
        result["service_type"]
        .map(PROMISED_HOURS)
        .astype("float64")
    )

    # Binary payment indicator.
    result["is_cod"] = (
        result["payment_mode"]
        .eq("COD")
        .astype("int64")
    )

    # Dispatch calendar features.
    result["is_weekend"] = (
        result["dispatch_time"]
        .dt.dayofweek
        .ge(5)
        .astype("int64")
    )

    result["dispatch_hour"] = (
        result["dispatch_time"]
        .dt.hour
        .astype("int64")
    )

    # Log-transform declared value. Clipping protects against
    # zero or negative values before taking the logarithm.
    result["log_declared_value"] = np.log(
        result["declared_value_inr"].clip(lower=1)
    )

    return result
