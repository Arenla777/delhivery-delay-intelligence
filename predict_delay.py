
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.shipment_prep import clean_shipments, build_features, FEATURES


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "models"
OUTPUT_DIR = ROOT / "outputs"

NEW_SHIPMENTS_PATH = DATA_DIR / "new_shipments.csv"
HUBS_PATH = DATA_DIR / "hubs.csv"
MODEL_PATH = MODEL_DIR / "delay_model.pkl"
THRESHOLD_PATH = MODEL_DIR / "delay_threshold.csv"
FEATURE_ORDER_PATH = MODEL_DIR / "feature_order.csv"
IMPUTE_VALUES_PATH = MODEL_DIR / "impute_values.csv"
SEGMENTS_PATH = OUTPUT_DIR / "hub_segments.csv"
OUTPUT_PATH = OUTPUT_DIR / "new_shipment_actions.csv"


def main():
    required_files = [
        NEW_SHIPMENTS_PATH,
        HUBS_PATH,
        MODEL_PATH,
        THRESHOLD_PATH,
        FEATURE_ORDER_PATH,
        IMPUTE_VALUES_PATH,
        SEGMENTS_PATH,
    ]

    for path in required_files:
        if not path.exists():
            raise FileNotFoundError(
                f"Required file not found: {path}\n"
                "Run the earlier project scripts first."
            )

    # Load input data, model and deployment settings.
    new_shipments = pd.read_csv(NEW_SHIPMENTS_PATH)
    hubs = pd.read_csv(HUBS_PATH)
    model = joblib.load(MODEL_PATH)

    threshold_df = pd.read_csv(THRESHOLD_PATH)
    feature_order_df = pd.read_csv(FEATURE_ORDER_PATH)
    impute_df = pd.read_csv(IMPUTE_VALUES_PATH)
    hub_segments = pd.read_csv(SEGMENTS_PATH)

    if len(threshold_df) != 1 or "threshold" not in threshold_df.columns:
        raise ValueError("delay_threshold.csv must contain one threshold value.")

    threshold = float(threshold_df.loc[0, "threshold"])

    if not 0 <= threshold <= 1:
        raise ValueError(f"Invalid threshold: {threshold}")

    if "feature" not in feature_order_df.columns:
        raise ValueError("feature_order.csv must contain a feature column.")

    feature_order = feature_order_df["feature"].astype(str).tolist()

    if feature_order != FEATURES:
        raise ValueError(
            "feature_order.csv does not match FEATURES in src/shipment_prep.py."
        )

    if not {"column", "median"}.issubset(impute_df.columns):
        raise ValueError("impute_values.csv must contain column and median.")

    medians = dict(
        zip(
            impute_df["column"].astype(str),
            pd.to_numeric(impute_df["median"], errors="raise"),
        )
    )

    expected_medians = {
        "distance_km",
        "parcel_weight_kg",
        "declared_value_inr",
    }

    if set(medians) != expected_medians:
        raise ValueError(
            f"Expected imputation medians for {sorted(expected_medians)}; "
            f"found {sorted(medians)}."
        )

    if not {"hub_id", "segment"}.issubset(hub_segments.columns):
        raise ValueError("hub_segments.csv must contain hub_id and segment.")

    if hub_segments["hub_id"].duplicated().any():
        raise ValueError("hub_segments.csv contains duplicate hub IDs.")

    if new_shipments["shipment_id"].duplicated().any():
        raise ValueError("new_shipments.csv contains duplicate shipment IDs.")

    original_count = len(new_shipments)
    original_order = new_shipments["shipment_id"].tolist()

    # Use the shared cleaning and feature-building functions.
    clean, _ = clean_shipments(new_shipments, medians=medians)
    features = build_features(clean, hubs)

    if len(features) != original_count:
        raise ValueError(
            f"Row count changed during preprocessing: "
            f"{original_count} input rows, {len(features)} feature rows."
        )

    if features["shipment_id"].tolist() != original_order:
        raise ValueError("Shipment order changed during preprocessing.")

    X_new = features[feature_order].copy()

    if X_new.isna().any().any():
        missing = X_new.columns[X_new.isna().any()].tolist()
        raise ValueError(f"Missing model features after preprocessing: {missing}")

    if not np.isfinite(X_new.to_numpy(dtype=float)).all():
        raise ValueError("Model features contain infinite or non-finite values.")

    # Positive-class probability means probability of a delayed shipment.
    delay_probabilities = model.predict_proba(X_new)[:, 1]
    rounded_probabilities = np.round(delay_probabilities, 4)

    # Apply the threshold to the unrounded probability.
    flagged = (delay_probabilities >= threshold).astype(int)

    scored = pd.DataFrame(
        {
            "shipment_id": features["shipment_id"].to_numpy(),
            "hub_id": features["hub_id"].to_numpy(),
            "delay_probability": rounded_probabilities,
            "flagged": flagged,
        }
    )

    scored = scored.merge(
        hub_segments[["hub_id", "segment"]],
        on="hub_id",
        how="left",
        validate="many_to_one",
        sort=False,
    )

    if scored["segment"].isna().any():
        missing_hubs = scored.loc[
            scored["segment"].isna(), "hub_id"
        ].unique().tolist()
        raise ValueError(f"No segment found for hubs: {missing_hubs}")

    scored["action"] = np.select(
        [
            (scored["flagged"] == 1) & (scored["segment"] == "Escalate"),
            scored["flagged"] == 1,
        ],
        [
            "Escalate to hub manager",
            "Proactive customer call",
        ],
        default="No action",
    )

    output_columns = [
        "shipment_id",
        "hub_id",
        "segment",
        "delay_probability",
        "flagged",
        "action",
    ]

    result = scored[output_columns].copy()

    if len(result) != 48:
        raise ValueError(f"Expected 48 scored shipments, found {len(result)}.")

    if result.isna().any().any():
        raise ValueError("Final action list contains missing values.")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Fixed column order and line endings make repeat runs reproducible.
    result.to_csv(
        OUTPUT_PATH,
        index=False,
        lineterminator="\n",
        float_format="%.4f",
    )

    flagged_count = int(result["flagged"].sum())
    manager_escalations = int(
        (result["action"] == "Escalate to hub manager").sum()
    )
    customer_calls = int(
        (result["action"] == "Proactive customer call").sum()
    )

    print("NEW SHIPMENT SCORING COMPLETE")
    print("------------------------------------")
    print(f"New shipments scored: {len(result)}")
    print(f"Alert threshold: {threshold:.4f}")
    print(f"Flagged shipments: {flagged_count}")
    print(f"Hub-manager escalations: {manager_escalations}")
    print(f"Proactive customer calls: {customer_calls}")
    print(f"No-action shipments: {len(result) - flagged_count}")

    print("\nACTION COUNTS")
    print(result["action"].value_counts().to_string())

    print("\nFIRST 10 SCORED SHIPMENTS")
    print(result.head(10).to_string(index=False))

    print(f"\nCreated: {OUTPUT_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
