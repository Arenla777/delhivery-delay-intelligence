
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from src.shipment_prep import FEATURES


ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "modelling_table.csv"
CONFIG_PATH = ROOT / "outputs" / "part_c_final_config.csv"
MODEL_DIR = ROOT / "models"
OUTPUT_DIR = ROOT / "outputs"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
TEST_SIZE = 0.15
VALIDATION_SIZE = 0.15


def main():
    for path in [DATA_PATH, CONFIG_PATH]:
        if not path.exists():
            raise FileNotFoundError(f"Required file not found: {path}")

    df = pd.read_csv(DATA_PATH)
    config = pd.read_csv(CONFIG_PATH)

    if len(config) != 1:
        raise ValueError("Expected exactly one row in part_c_final_config.csv.")

    settings = config.iloc[0]
    selector_k = int(settings["selector_k"])
    n_estimators = int(settings["n_estimators"])
    max_depth_text = str(settings["max_depth"]).strip()
    min_samples_leaf = int(settings["min_samples_leaf"])
    threshold = float(settings["threshold"])

    max_depth = (
        None
        if max_depth_text.lower() in {"unlimited", "none", "nan"}
        else int(float(max_depth_text))
    )

    required_columns = FEATURES + ["is_delayed", "shipment_id"]
    missing = [column for column in required_columns if column not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    X = df[FEATURES].copy()
    y = df["is_delayed"].astype(int).copy()

    # Rebuild the same 70/15/15 stratified split used in Part C.
    X_train, X_temp, y_train, y_temp = train_test_split(
        X,
        y,
        test_size=0.30,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=0.50,
        random_state=RANDOM_STATE,
        stratify=y_temp,
    )

    if len(X_train) != 840 or len(X_val) != 180 or len(X_test) != 180:
        raise ValueError(
            f"Unexpected split sizes: train={len(X_train)}, "
            f"validation={len(X_val)}, test={len(X_test)}"
        )

    if not 1 <= selector_k <= len(FEATURES):
        raise ValueError(f"Invalid selector_k: {selector_k}")

    model = Pipeline(
        steps=[
            ("selector", SelectKBest(score_func=f_classif, k=selector_k)),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    min_samples_leaf=min_samples_leaf,
                    random_state=RANDOM_STATE,
                    n_jobs=1,
                ),
            ),
        ]
    )

    # Fit on training rows only. Validation/test rows are not used for fitting.
    model.fit(X_train, y_train)

    model_path = MODEL_DIR / "delay_model.pkl"
    joblib.dump(model, model_path)

    pd.DataFrame({"threshold": [threshold]}).to_csv(
        MODEL_DIR / "delay_threshold.csv", index=False
    )

    pd.DataFrame({"feature": FEATURES}).to_csv(
        MODEL_DIR / "feature_order.csv", index=False
    )

    # Recalculate imputation medians from the cleaned training source table.
    # These are recorded for transparent deployment documentation.
    raw_columns = ["distance_km", "parcel_weight_kg", "declared_value_inr"]
    medians = {}

    for column in raw_columns:
        if column not in df.columns:
            raise ValueError(
                f"{column} is missing from modelling_table.csv; "
                "cannot create impute_values.csv."
            )

        medians[column] = float(pd.to_numeric(df[column], errors="coerce").median())

    pd.DataFrame(
        [
            {"column": column, "median": value}
            for column, value in medians.items()
        ]
    ).to_csv(MODEL_DIR / "impute_values.csv", index=False)

    # Reload the saved pipeline and compare its test probabilities with the
    # in-memory model's probabilities.
    reloaded_model = joblib.load(model_path)

    in_memory_probabilities = model.predict_proba(X_test)[:, 1]
    reloaded_probabilities = reloaded_model.predict_proba(X_test)[:, 1]

    max_difference = float(
        np.max(np.abs(in_memory_probabilities - reloaded_probabilities))
    )

    if max_difference > 1e-12:
        raise AssertionError(
            f"Reloaded model predictions differ: {max_difference}"
        )

    pd.DataFrame(
        [
            {"metric": "test_rows_scored", "value": len(X_test)},
            {
                "metric": "reload_max_abs_probability_diff",
                "value": max_difference,
            },
            {"metric": "threshold", "value": threshold},
        ]
    ).to_csv(OUTPUT_DIR / "part_d_reload_check.csv", index=False)

    print("MODEL PACKAGING COMPLETE")
    print("------------------------------------")
    print(f"Training rows:   {len(X_train)}")
    print(f"Validation rows: {len(X_val)}")
    print(f"Test rows:       {len(X_test)}")
    print(f"Selected features (k): {selector_k}")
    print(f"Random forest trees: {n_estimators}")
    print(f"Maximum depth: {max_depth}")
    print(f"Alert threshold: {threshold}")
    print(f"Reload max probability difference: {max_difference:.12f}")

    print("\nCreated:")
    print(" - models/delay_model.pkl")
    print(" - models/delay_threshold.csv")
    print(" - models/feature_order.csv")
    print(" - models/impute_values.csv")
    print(" - outputs/part_d_reload_check.csv")


if __name__ == "__main__":
    main()
