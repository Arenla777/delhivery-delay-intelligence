# Delhivery Last-Mile Delay Intelligence Pipeline

## Overview

This project develops a reproducible machine-learning pipeline to analyze shipment delays, predict delivery performance, segment operational hubs, and recommend actions for newly received shipments.

The pipeline covers data cleaning, exploratory analysis, delivery-time regression, delay classification, hub segmentation, model packaging, and operational scoring.

## Project Workflow

1. Generate and prepare shipment data.
2. Clean records and construct modelling features.
3. Analyze shipment delays across service types, hub tiers, and distance bands.
4. Train and evaluate delivery-time regression models.
5. Train and evaluate a shipment-delay classifier.
6. Segment operational hubs using clustering.
7. Package the selected model and score new shipments.
8. Generate action recommendations for flagged shipments.

## Key Data Findings

- Raw shipment records: 1,224
- Duplicate records removed: 24
- Clean shipment records: 1,200
- Delayed shipments: 278
- Overall delay rate: 23.17%
- Model features: 17

Delay rates varied considerably by operating conditions:

| Group | Delay rate |
|---|---:|
| Economy service | 8.57% |
| Standard service | 20.56% |
| Express service | 48.97% |
| Tier 1 hubs | 4.00% |
| Tier 2 hubs | 21.50% |
| Tier 3 hubs | 44.00% |
| Distance 0–200 km | 2.19% |
| Distance 201–400 km | 13.68% |
| Distance 401–600 km | 43.64% |
| Distance above 600 km | 85.23% |

These results indicate that service type, hub tier, and delivery distance are associated with substantial differences in observed delay rates.

## Regression Results

The final linear regression model achieved the following test-set performance:

| Metric | Result |
|---|---:|
| Mean Absolute Error (MAE) | 3.5671 |
| Root Mean Squared Error (RMSE) | 4.5235 |
| R² | 0.9493 |

The cross-validation mean R² for linear regression was 0.9261. Gradient boosting achieved a cross-validation mean R² of 0.9353.

## Delay Classification

The final Random Forest classifier used 7 selected features and 100 trees, with a classification threshold of 0.50.

| Metric | Test result |
|---|---:|
| Accuracy | 86.67% |
| Precision | 68.89% |
| Recall | 75.61% |
| F1 score | 72.09% |

Confusion matrix:

| | Predicted on time | Predicted delayed |
|---|---:|---:|
| Actually on time | 125 | 14 |
| Actually delayed | 10 | 31 |

The classifier identifies a subset of shipments at elevated risk of delay. Its predictions should support, rather than replace, operational judgement.

## Hub Segmentation

The hub segmentation analysis grouped 24 hubs into three operational categories.

| Segment | Hubs | Mean delivery time (hours) | Delay rate |
|---|---:|---:|---:|
| Standard | 7 | 47.24 | 2.57% |
| Watch | 8 | 56.81 | 15.50% |
| Escalate | 9 | 64.65 | 46.00% |

Four hubs differed from the expected category based on their hub tier. These cases warrant further investigation rather than automatic reclassification.

## New Shipment Scoring

The scoring pipeline processed 48 new shipments using the packaged model and a probability threshold of 0.50.

| Recommended action | Shipments |
|---|---:|
| Escalate to hub manager | 6 |
| Proactive customer call | 2 |
| No action | 40 |

The resulting shipment-level recommendations are saved in `outputs/new_shipment_actions.csv`.

## Reproducibility and Model Packaging

The packaged model and supporting configuration are saved under `models/`. The scoring script was run twice, and both runs produced the same SHA-256 hash for the output CSV, confirming identical file contents across those runs.

## Repository Structure

```text
delhivery-delay-intelligence/
├── data/
│   ├── hubs.csv
│   ├── raw_shipments.csv
│   ├── new_shipments.csv
│   └── modelling_table.csv
├── src/
│   └── shipment_prep.py
├── models/
│   ├── delay_model.pkl
│   ├── delay_threshold.csv
│   ├── feature_order.csv
│   └── impute_values.csv
├── outputs/
│   ├── hub_profile.csv
│   ├── part_d_separation.csv
│   ├── hub_segments.csv
│   ├── segment_profile.csv
│   ├── tier_vs_segment_mismatch.csv
│   ├── part_d_reload_check.csv
│   └── new_shipment_actions.csv
├── generate_data.py
├── part_a_clean_eda.py
├── part_b_regression.py
├── part_c_classification.py
├── part_d_segmentation.py
├── package_model.py
├── predict_delay.py
├── requirements.txt
└── README.md
```

## Limitations

- The generated dataset is suitable for demonstrating the pipeline, not for claiming validated performance on real Delhivery operations.
- Model performance on new, real-world shipments may differ from the reported test results.
- Classification thresholds and recommended actions should be evaluated against operational costs, customer impact, and false-negative risk before deployment.
- Hub categories and shipment-level recommendations require monitoring and periodic reassessment.

## Technology Stack

- Python
- pandas and NumPy
- scikit-learn
- joblib

## Running the Pipeline

Run the scripts from the repository root using the Python interpreter configured on your machine. Install dependencies first using:

```powershell
python -m pip install -r requirements.txt
```

Then run the data-generation, analysis, modelling, packaging, and scoring scripts in the order required by the project workflow.