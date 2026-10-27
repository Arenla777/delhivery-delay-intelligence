
from pathlib import Path
from itertools import product

import numpy as np
import pandas as pd

from sklearn.model_selection import (
    train_test_split,
    GridSearchCV,
    RandomizedSearchCV,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)


# ---------------------------------------------------------
# 1. Paths and assignment configuration
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

NOISE_COLUMNS = [
    "label_print_batch",
    "scanner_build_no",
    "tape_roll_no",
    "sticker_serial_mod",
    "shelf_slot_no",
]

TARGET = "is_delayed"
RANDOM_STATE = 42


# ---------------------------------------------------------
# 2. Load and validate the modelling table
# ---------------------------------------------------------

if not DATA_PATH.exists():
    raise FileNotFoundError(
        f"Cannot find {DATA_PATH}. Run part_a_clean_eda.py first."
    )

df = pd.read_csv(DATA_PATH)

required_columns = FEATURES + [TARGET]
missing_columns = [c for c in required_columns if c not in df.columns]

if missing_columns:
    raise ValueError(f"Missing required columns: {missing_columns}")

if df[required_columns].isnull().any().any():
    raise ValueError(
        "Missing values found in features or target. "
        "Check Part A before continuing."
    )

X = df[FEATURES].astype(float)
y = df[TARGET].astype(int)


# ---------------------------------------------------------
# 3. Stratified train / validation / test split
# ---------------------------------------------------------

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

print("\nSTRATIFIED DATA SPLIT")
print(f"Training rows:   {len(X_train)}")
print(f"Validation rows: {len(X_val)}")
print(f"Test rows:       {len(X_test)}")
print(
    "Delayed in train / validation / test:",
    int(y_train.sum()),
    int(y_val.sum()),
    int(y_test.sum()),
)


# ---------------------------------------------------------
# 4. Metric helper
# ---------------------------------------------------------

def evaluate_predictions(model_name, actual, predicted):
    tn, fp, fn, tp = confusion_matrix(
        actual, predicted, labels=[0, 1]
    ).ravel()

    return {
        "model": model_name,
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "TP": int(tp),
        "accuracy": accuracy_score(actual, predicted),
        "precision": precision_score(
            actual, predicted, zero_division=0
        ),
        "recall": recall_score(
            actual, predicted, zero_division=0
        ),
        "f1": f1_score(actual, predicted, zero_division=0),
    }


# ---------------------------------------------------------
# 5. Blind baseline and validation classifiers
# ---------------------------------------------------------

model_results = []

blind_predictions = np.zeros(len(y_val), dtype=int)
blind_result = evaluate_predictions(
    "blind_all_on_time", y_val, blind_predictions
)
# The assignment explicitly requires blind-baseline precision = 0.
blind_result["precision"] = 0.0
model_results.append(blind_result)

logistic_model = Pipeline(
    [
        ("scaler", StandardScaler()),
        ("model", LogisticRegression()),
    ]
)
logistic_model.fit(X_train, y_train)

model_results.append(
    evaluate_predictions(
        "logistic_regression",
        y_val,
        logistic_model.predict(X_val),
    )
)


# ---------------------------------------------------------
# 6. Choose decision-tree depth by validation F1
# ---------------------------------------------------------

tree_depth_results = []

for depth in [2, 3, 4, 5, 6, 8, 10]:
    tree = DecisionTreeClassifier(
        criterion="gini",
        max_depth=depth,
        random_state=RANDOM_STATE,
    )
    tree.fit(X_train, y_train)
    predictions = tree.predict(X_val)

    tree_depth_results.append(
        {
            "depth": depth,
            "f1": f1_score(y_val, predictions, zero_division=0),
        }
    )

best_tree_row = sorted(
    tree_depth_results,
    key=lambda row: (-row["f1"], row["depth"]),
)[0]
best_tree_depth = best_tree_row["depth"]

best_tree = DecisionTreeClassifier(
    criterion="gini",
    max_depth=best_tree_depth,
    random_state=RANDOM_STATE,
)
best_tree.fit(X_train, y_train)

model_results.append(
    evaluate_predictions(
        f"decision_tree_depth_{best_tree_depth}",
        y_val,
        best_tree.predict(X_val),
    )
)

default_forest = RandomForestClassifier(
    n_estimators=100,
    random_state=RANDOM_STATE,
)
default_forest.fit(X_train, y_train)

model_results.append(
    evaluate_predictions(
        "random_forest_default",
        y_val,
        default_forest.predict(X_val),
    )
)

pd.DataFrame(model_results).to_csv(
    OUTPUT_DIR / "part_c_models.csv",
    index=False,
    float_format="%.4f",
)

print("\nVALIDATION CLASSIFIER RESULTS")
print(
    pd.DataFrame(model_results)
    .round(4)
    .to_string(index=False)
)
print(f"\nSelected decision-tree depth: {best_tree_depth}")


# ---------------------------------------------------------
# 7. Feature F-scores and random-forest importance
# ---------------------------------------------------------

importance_forest = RandomForestClassifier(
    n_estimators=100,
    random_state=RANDOM_STATE,
)
importance_forest.fit(X_train, y_train)

selector_all = SelectKBest(score_func=f_classif, k="all")
selector_all.fit(X_train, y_train)

f_scores = np.nan_to_num(
    selector_all.scores_,
    nan=0.0,
    posinf=0.0,
    neginf=0.0,
)

# Rank 1 means the highest F-score.
f_ranks = (
    pd.Series(f_scores)
    .rank(method="min", ascending=False)
    .astype(int)
    .to_numpy()
)

feature_scores = pd.DataFrame(
    {
        "feature": FEATURES,
        "f_score": f_scores,
        "f_rank": f_ranks,
        "rf_importance": importance_forest.feature_importances_,
        "is_noise": [feature in NOISE_COLUMNS for feature in FEATURES],
    }
)

feature_scores.to_csv(
    OUTPUT_DIR / "part_c_feature_scores.csv",
    index=False,
    float_format="%.4f",
)

noise_share = feature_scores.loc[
    feature_scores["is_noise"], "rf_importance"
].sum()

print(f"\nCombined noise-column forest importance: {noise_share:.4f}")


# ---------------------------------------------------------
# 8. SelectKBest: compare k = 5, 7, 10, 12, 17
# ---------------------------------------------------------

select_k_results = []

for k in [5, 7, 10, 12, 17]:
    model = Pipeline(
        [
            ("selector", SelectKBest(score_func=f_classif, k=k)),
            (
                "model",
                RandomForestClassifier(
                    n_estimators=100,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )

    model.fit(X_train, y_train)
    predictions = model.predict(X_val)

    select_k_results.append(
        {
            "k": k,
            "val_f1": f1_score(
                y_val, predictions, zero_division=0
            ),
        }
    )

pd.DataFrame(select_k_results).to_csv(
    OUTPUT_DIR / "part_c_selectk.csv",
    index=False,
    float_format="%.4f",
)

print("\nFEATURE-SELECTION RESULTS")
print(pd.DataFrame(select_k_results).round(4).to_string(index=False))


# ---------------------------------------------------------
# 9. Fit the k = 7 selector on training rows only
# ---------------------------------------------------------

selector_7 = SelectKBest(score_func=f_classif, k=7)
X_train_7 = selector_7.fit_transform(X_train, y_train)
X_val_7 = selector_7.transform(X_val)

parameter_grid = {
    "n_estimators": [50, 100, 150],
    "max_depth": [4, 6, 8],
    "min_samples_leaf": [1, 5, 10],
}

# Explicitly list the 27 combinations so the number of candidates
# and total CV fits are recorded accurately.
all_combinations = [
    {
        "n_estimators": n,
        "max_depth": d,
        "min_samples_leaf": leaf,
    }
    for n, d, leaf in product(
        parameter_grid["n_estimators"],
        parameter_grid["max_depth"],
        parameter_grid["min_samples_leaf"],
    )
]

grid_search = GridSearchCV(
    estimator=RandomForestClassifier(random_state=RANDOM_STATE),
    param_grid=parameter_grid,
    cv=5,
)
grid_search.fit(X_train_7, y_train)

random_search = RandomizedSearchCV(
    estimator=RandomForestClassifier(random_state=RANDOM_STATE),
    param_distributions=parameter_grid,
    n_iter=9,
    cv=5,
    random_state=RANDOM_STATE,
)
random_search.fit(X_train_7, y_train)

default_forest_7 = RandomForestClassifier(
    n_estimators=100,
    random_state=RANDOM_STATE,
    max_depth=None,
    min_samples_leaf=1,
)
default_forest_7.fit(X_train_7, y_train)

grid_val_predictions = grid_search.best_estimator_.predict(X_val_7)
random_val_predictions = random_search.best_estimator_.predict(X_val_7)
default_7_val_predictions = default_forest_7.predict(X_val_7)

tuning_results = [
    {
        "search": "grid_search",
        "candidates": 27,
        "cv_folds": 5,
        "total_fits": 27 * 5,
        "best_params": str(grid_search.best_params_),
        "best_cv_accuracy": grid_search.best_score_,
        "val_f1": f1_score(
            y_val, grid_val_predictions, zero_division=0
        ),
    },
    {
        "search": "randomized_search",
        "candidates": 9,
        "cv_folds": 5,
        "total_fits": 9 * 5,
        "best_params": str(random_search.best_params_),
        "best_cv_accuracy": random_search.best_score_,
        "val_f1": f1_score(
            y_val, random_val_predictions, zero_division=0
        ),
    },
    {
        "search": "default_forest_same_7_features",
        "candidates": 1,
        "cv_folds": 0,
        "total_fits": 1,
        "best_params": "defaults",
        "best_cv_accuracy": np.nan,
        "val_f1": f1_score(
            y_val, default_7_val_predictions, zero_division=0
        ),
    },
]

tuning_df = pd.DataFrame(tuning_results)
tuning_df.to_csv(
    OUTPUT_DIR / "part_c_tuning.csv",
    index=False,
    float_format="%.4f",
)

print("\nTUNING RESULTS")
print(tuning_df.round(4).to_string(index=False))


# ---------------------------------------------------------
# 10. Choose final configuration by validation F1
# ---------------------------------------------------------

tie_priority = {
    "default_forest_same_7_features": 0,
    "grid_search": 1,
    "randomized_search": 2,
}

best_tuning_row = sorted(
    tuning_results,
    key=lambda row: (
        -row["val_f1"],
        tie_priority[row["search"]],
    ),
)[0]

chosen_from = best_tuning_row["search"]

if chosen_from == "grid_search":
    chosen_params = grid_search.best_params_
elif chosen_from == "randomized_search":
    chosen_params = random_search.best_params_
else:
    chosen_params = {
        "n_estimators": 100,
        "max_depth": None,
        "min_samples_leaf": 1,
    }

final_model = Pipeline(
    [
        ("selector", SelectKBest(score_func=f_classif, k=7)),
        (
            "model",
            RandomForestClassifier(
                random_state=RANDOM_STATE,
                **chosen_params,
            ),
        ),
    ]
)

final_model.fit(X_train, y_train)


# ---------------------------------------------------------
# 11. Choose alert threshold from validation recall
# ---------------------------------------------------------

val_probabilities = final_model.predict_proba(X_val)[:, 1]
threshold_values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

threshold_results = []

for threshold in threshold_values:
    predictions = (val_probabilities >= threshold).astype(int)
    result = evaluate_predictions(
        "threshold",
        y_val,
        predictions,
    )
    result["threshold"] = threshold
    threshold_results.append(result)

threshold_df = pd.DataFrame(threshold_results)

threshold_df = threshold_df[
    [
        "threshold",
        "TN",
        "FP",
        "FN",
        "TP",
        "accuracy",
        "precision",
        "recall",
        "f1",
    ]
]

threshold_df.to_csv(
    OUTPUT_DIR / "part_c_thresholds.csv",
    index=False,
    float_format="%.4f",
)

eligible_thresholds = threshold_df[
    threshold_df["recall"] >= 0.80
]

if eligible_thresholds.empty:
    chosen_threshold = 0.1
else:
    chosen_threshold = float(eligible_thresholds["threshold"].max())

print("\nTHRESHOLD COMPARISON")
print(threshold_df.round(4).to_string(index=False))
print(f"\nChosen alert threshold: {chosen_threshold:.1f}")


# ---------------------------------------------------------
# 12. Evaluate the test set exactly once at chosen threshold
# ---------------------------------------------------------

test_probabilities = final_model.predict_proba(X_test)[:, 1]
test_predictions = (test_probabilities >= chosen_threshold).astype(int)

final_test_result = evaluate_predictions(
    "final_model",
    y_test,
    test_predictions,
)

final_test_result.pop("model")
final_test_result = {
    "threshold": chosen_threshold,
    **final_test_result,
}

pd.DataFrame([final_test_result]).to_csv(
    OUTPUT_DIR / "part_c_final_test.csv",
    index=False,
    float_format="%.4f",
)


# ---------------------------------------------------------
# 13. Save the final configuration
# ---------------------------------------------------------

final_config = pd.DataFrame(
    [
        {
            "selector_k": 7,
            "n_estimators": chosen_params["n_estimators"],
            "max_depth": (
                "unlimited"
                if chosen_params["max_depth"] is None
                else chosen_params["max_depth"]
            ),
            "min_samples_leaf": chosen_params["min_samples_leaf"],
            "threshold": chosen_threshold,
            "chosen_from": chosen_from,
        }
    ]
)

final_config.to_csv(
    OUTPUT_DIR / "part_c_final_config.csv",
    index=False,
    float_format="%.4f",
)

print("\nFINAL TEST RESULT")
print(
    pd.DataFrame([final_test_result])
    .round(4)
    .to_string(index=False)
)

print("\nFINAL CONFIGURATION")
print(final_config.to_string(index=False))

print("\nPART C COMPLETE")
print("Created:")
print(" - part_c_models.csv")
print(" - part_c_feature_scores.csv")
print(" - part_c_selectk.csv")
print(" - part_c_tuning.csv")
print(" - part_c_thresholds.csv")
print(" - part_c_final_test.csv")
print(" - part_c_final_config.csv")
