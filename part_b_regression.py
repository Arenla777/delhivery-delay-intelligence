
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ---------------------------------------------------------
# 1. Paths and configuration
# ---------------------------------------------------------

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "modelling_table.csv"
OUTPUT_DIR = ROOT / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

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

TARGET = "delivery_hours"
RANDOM_STATE = 42


# ---------------------------------------------------------
# 2. Load the modelling table and check the data
# ---------------------------------------------------------

if not DATA_PATH.exists():
    raise FileNotFoundError(
        f"Cannot find {DATA_PATH}. Run part_a_clean_eda.py first."
    )

df = pd.read_csv(DATA_PATH)

required_columns = FEATURES + [TARGET]
missing_columns = [col for col in required_columns if col not in df.columns]

if missing_columns:
    raise ValueError(f"Missing required columns: {missing_columns}")

if df[required_columns].isnull().any().any():
    raise ValueError(
        "Missing values detected in features or target. "
        "Check Part A before continuing."
    )

X = df[FEATURES].astype(float)
y = df[TARGET].astype(float)


# ---------------------------------------------------------
# 3. Split into train, validation and test sets
# ---------------------------------------------------------

X_train, X_temp, y_train, y_temp = train_test_split(
    X,
    y,
    test_size=0.30,
    random_state=RANDOM_STATE,
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp,
    y_temp,
    test_size=0.50,
    random_state=RANDOM_STATE,
)

print("\nDATA SPLIT")
print(f"Training rows:   {len(X_train)}")
print(f"Validation rows: {len(X_val)}")
print(f"Test rows:       {len(X_test)}")


# ---------------------------------------------------------
# 4. Check scaling: fit the scaler on training data only
# ---------------------------------------------------------

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

train_scaled_max_abs_mean = float(
    np.abs(X_train_scaled.mean(axis=0)).max()
)

val_scaled_max_abs_mean = float(
    np.abs(X_val_scaled.mean(axis=0)).max()
)

test_scaled_max_abs_mean = float(
    np.abs(X_test_scaled.mean(axis=0)).max()
)

checks = pd.DataFrame(
    [
        {"check": "train_rows", "value": len(X_train)},
        {"check": "validation_rows", "value": len(X_val)},
        {"check": "test_rows", "value": len(X_test)},
        {
            "check": "train_scaled_max_abs_mean",
            "value": train_scaled_max_abs_mean,
        },
        {
            "check": "validation_scaled_max_abs_mean",
            "value": val_scaled_max_abs_mean,
        },
        {
            "check": "test_scaled_max_abs_mean",
            "value": test_scaled_max_abs_mean,
        },
    ]
)

checks.to_csv(OUTPUT_DIR / "part_b_checks.csv", index=False)


# ---------------------------------------------------------
# 5. Evaluation helper
# ---------------------------------------------------------

def calculate_metrics(model_name, split_name, actual, predicted):
    mse = mean_squared_error(actual, predicted)

    return {
        "model": model_name,
        "split": split_name,
        "MAE": mean_absolute_error(actual, predicted),
        "MSE": mse,
        "RMSE": np.sqrt(mse),
        "R2": r2_score(actual, predicted),
    }


# ---------------------------------------------------------
# 6. Baseline model: predict the training mean
# ---------------------------------------------------------

baseline_prediction = np.full(len(y_val), y_train.mean())

validation_results = [
    calculate_metrics(
        "mean_baseline", "val", y_val, baseline_prediction
    )
]


# ---------------------------------------------------------
# 7. Diagnose decision-tree depth
# ---------------------------------------------------------

depth_values = [1, 2, 3, 4, 6, 8, 12, None]
depth_results = []

for depth in depth_values:
    tree = DecisionTreeRegressor(
        max_depth=depth,
        random_state=RANDOM_STATE,
    )

    tree.fit(X_train, y_train)

    train_prediction = tree.predict(X_train)
    val_prediction = tree.predict(X_val)

    train_rmse = np.sqrt(
        mean_squared_error(y_train, train_prediction)
    )
    val_rmse = np.sqrt(
        mean_squared_error(y_val, val_prediction)
    )

    depth_results.append(
        {
            "max_depth": "unlimited" if depth is None else depth,
            "train_rmse": train_rmse,
            "val_rmse": val_rmse,
        }
    )

depth_df = pd.DataFrame(depth_results)
depth_df.to_csv(
    OUTPUT_DIR / "part_b_tree_depth.csv",
    index=False,
    float_format="%.4f",
)

# Select the depth with the lowest validation RMSE.
# In a tie, prefer the shallower tree.
best_depth_row = min(
    depth_results,
    key=lambda row: (
        row["val_rmse"],
        float("inf")
        if row["max_depth"] == "unlimited"
        else row["max_depth"],
    ),
)

best_depth = (
    None
    if best_depth_row["max_depth"] == "unlimited"
    else int(best_depth_row["max_depth"])
)

print("\nDECISION TREE DEPTH DIAGNOSTIC")
print(depth_df.to_string(index=False))
print(f"\nSelected tree depth: {best_depth}")


# ---------------------------------------------------------
# 8. Train the four regression models
# ---------------------------------------------------------

models = {
    "linear_regression": Pipeline(
        [
            ("scaler", StandardScaler()),
            ("regressor", LinearRegression()),
        ]
    ),
    "decision_tree": DecisionTreeRegressor(
        max_depth=best_depth,
        random_state=RANDOM_STATE,
    ),
    "random_forest": RandomForestRegressor(
        n_estimators=100,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    ),
    "gradient_boosting": GradientBoostingRegressor(
        n_estimators=100,
        learning_rate=0.1,
        random_state=RANDOM_STATE,
    ),
}

for name, model in models.items():
    model.fit(X_train, y_train)
    predictions = model.predict(X_val)

    validation_results.append(
        calculate_metrics(name, "val", y_val, predictions)
    )


# ---------------------------------------------------------
# 9. Compare Gradient Boosting learning rates
# ---------------------------------------------------------

learning_rate_results = []

for learning_rate in [0.01, 0.1, 1.0]:
    model = GradientBoostingRegressor(
        n_estimators=100,
        learning_rate=learning_rate,
        random_state=RANDOM_STATE,
    )

    model.fit(X_train, y_train)
    predictions = model.predict(X_val)

    metrics = calculate_metrics(
        f"gradient_boosting_lr_{learning_rate}",
        "val",
        y_val,
        predictions,
    )

    learning_rate_results.append(
        {
            "learning_rate": learning_rate,
            "val_rmse": metrics["RMSE"],
            "val_r2": metrics["R2"],
        }
    )

pd.DataFrame(learning_rate_results).to_csv(
    OUTPUT_DIR / "part_b_gb_learning_rate.csv",
    index=False,
    float_format="%.4f",
)


# ---------------------------------------------------------
# 10. Save validation metrics and choose the best model
# ---------------------------------------------------------

metrics_df = pd.DataFrame(validation_results)
metrics_df = metrics_df.sort_values("RMSE").reset_index(drop=True)

# Exclude the baseline when choosing the final predictive model.
candidate_metrics = metrics_df[
    metrics_df["model"] != "mean_baseline"
]

best_model_name = candidate_metrics.iloc[0]["model"]
best_model = models[best_model_name]

print("\nVALIDATION METRICS")
print(metrics_df.round(4).to_string(index=False))
print(f"\nBest model by validation RMSE: {best_model_name}")


# ---------------------------------------------------------
# 11. Five-fold cross-validation for the two best models
# ---------------------------------------------------------

# Use training + validation rows only. Keep the test set untouched.
X_train_val = pd.concat([X_train, X_val], axis=0)
y_train_val = pd.concat([y_train, y_val], axis=0)

top_two_names = candidate_metrics.sort_values("RMSE")["model"].head(2).tolist()

cv = KFold(
    n_splits=5,
    shuffle=True,
    random_state=RANDOM_STATE,
)

cv_results = []

for name in top_two_names:
    model = models[name]

    scores = cross_val_score(
        model,
        X_train_val,
        y_train_val,
        cv=cv,
        scoring="r2",
        n_jobs=-1,
    )

    cv_results.append(
        {
            "model": name,
            "mean_r2": scores.mean(),
            "std_r2": scores.std(),
        }
    )

pd.DataFrame(cv_results).to_csv(
    OUTPUT_DIR / "part_b_cv.csv",
    index=False,
    float_format="%.4f",
)


# ---------------------------------------------------------
# 12. Evaluate the selected model on the test set once
# ---------------------------------------------------------

test_predictions = best_model.predict(X_test)

test_result = calculate_metrics(
    best_model_name,
    "test",
    y_test,
    test_predictions,
)

metrics_df = pd.concat(
    [metrics_df, pd.DataFrame([test_result])],
    ignore_index=True,
)

metrics_df.to_csv(
    OUTPUT_DIR / "part_b_metrics.csv",
    index=False,
    float_format="%.4f",
)


# ---------------------------------------------------------
# 13. Print final results
# ---------------------------------------------------------

print("\nGRADIENT BOOSTING LEARNING-RATE RESULTS")
print(
    pd.read_csv(OUTPUT_DIR / "part_b_gb_learning_rate.csv")
    .round(4)
    .to_string(index=False)
)

print("\nCROSS-VALIDATION RESULTS")
print(
    pd.read_csv(OUTPUT_DIR / "part_b_cv.csv")
    .round(4)
    .to_string(index=False)
)

print("\nFINAL TEST RESULT")
print(pd.DataFrame([test_result]).round(4).to_string(index=False))

print("\nPART B COMPLETE")
print(f"Output folder: {OUTPUT_DIR}")
print("Created:")
print(" - part_b_checks.csv")
print(" - part_b_tree_depth.csv")
print(" - part_b_gb_learning_rate.csv")
print(" - part_b_cv.csv")
print(" - part_b_metrics.csv")
