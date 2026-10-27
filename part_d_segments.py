
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import pdist


# ---------------------------------------------------------
# 1. Paths and configuration
# ---------------------------------------------------------

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "modelling_table.csv"
OUTPUT_DIR = ROOT / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PROFILE_FEATURES = [
    "mean_delivery_hours",
    "delay_rate",
    "mean_distance_km",
    "mean_n_stops",
]

SEGMENT_NAMES = ["Standard", "Watch", "Escalate"]

TIER_TO_SEGMENT = {
    1: "Standard",
    2: "Watch",
    3: "Escalate",
}


# ---------------------------------------------------------
# 2. Load modelling table
# ---------------------------------------------------------

if not DATA_PATH.exists():
    raise FileNotFoundError(
        "data/modelling_table.csv not found. "
        "Run part_a_clean_eda.py first."
    )

df = pd.read_csv(DATA_PATH)

required = [
    "hub_id",
    "hub_city",
    "hub_tier",
    "delivery_hours",
    "is_delayed",
    "distance_km",
    "n_stops",
]

missing = [column for column in required if column not in df.columns]

if missing:
    raise ValueError(f"Missing required columns: {missing}")


# ---------------------------------------------------------
# 3. Create hub-level profile
# ---------------------------------------------------------

hub_profile = (
    df.groupby("hub_id", as_index=False)
    .agg(
        shipments=("shipment_id", "count"),
        mean_delivery_hours=("delivery_hours", "mean"),
        delay_rate=("is_delayed", "mean"),
        mean_distance_km=("distance_km", "mean"),
        mean_n_stops=("n_stops", "mean"),
    )
    .sort_values("hub_id")
    .reset_index(drop=True)
)

# Add city and tier for later segmentation outputs.
hub_metadata = (
    df[["hub_id", "hub_city", "hub_tier"]]
    .drop_duplicates()
)

if hub_metadata["hub_id"].duplicated().any():
    raise ValueError("A hub has inconsistent city or tier metadata.")

hub_profile = hub_profile.merge(
    hub_metadata,
    on="hub_id",
    how="left",
    validate="one_to_one",
)

# Save the required profile columns in the specified order.
hub_profile_output = hub_profile[
    [
        "hub_id",
        "shipments",
        "mean_delivery_hours",
        "delay_rate",
        "mean_distance_km",
        "mean_n_stops",
    ]
].copy()

hub_profile_output.to_csv(
    OUTPUT_DIR / "hub_profile.csv",
    index=False,
    float_format="%.4f",
)

if len(hub_profile) != 24:
    raise ValueError(
        f"Expected 24 hubs, but found {len(hub_profile)}."
    )

print("HUB PROFILE")
print(f"Number of hubs: {len(hub_profile)}")
print(hub_profile_output.head().round(4).to_string(index=False))


# ---------------------------------------------------------
# 4. Scale hub profiles and fit the required K-Means model
# ---------------------------------------------------------

profile_matrix = hub_profile[PROFILE_FEATURES].to_numpy(dtype=float)

scaler = StandardScaler()
scaled_profiles = scaler.fit_transform(profile_matrix)

kmeans = KMeans(
    n_clusters=3,
    random_state=42,
    n_init=10,
)

cluster_labels = kmeans.fit_predict(scaled_profiles)

hub_profile["cluster_id"] = cluster_labels


# ---------------------------------------------------------
# 5. Calculate across-to-within distance ratios
# ---------------------------------------------------------

# pdist returns Euclidean distances for all unordered pairs.
all_pair_distances = pdist(scaled_profiles, metric="euclidean")

# Build matching cluster labels for those same unordered pairs.
pair_cluster_labels = []
for i in range(len(cluster_labels)):
    for j in range(i + 1, len(cluster_labels)):
        pair_cluster_labels.append(
            cluster_labels[i] == cluster_labels[j]
        )

same_cluster = np.asarray(pair_cluster_labels, dtype=bool)
different_cluster = ~same_cluster

if len(all_pair_distances) != 276:
    raise ValueError(
        f"Expected 276 hub pairs, got {len(all_pair_distances)}."
    )

separation_results = []

for k in [2, 3, 4, 5]:
    candidate = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10,
    )
    labels = candidate.fit_predict(scaled_profiles)

    pair_same = []
    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            pair_same.append(labels[i] == labels[j])

    pair_same = np.asarray(pair_same, dtype=bool)
    pair_different = ~pair_same

    mean_within = all_pair_distances[pair_same].mean()
    mean_across = all_pair_distances[pair_different].mean()

    ratio = (
        mean_across / mean_within
        if mean_within > 0
        else np.nan
    )

    sizes = sorted(
        pd.Series(labels).value_counts().tolist(),
        reverse=True,
    )

    separation_results.append(
        {
            "K": k,
            "across_over_within": ratio,
            "cluster_sizes": "-".join(map(str, sizes)),
        }
    )

separation_df = pd.DataFrame(separation_results)

separation_df.to_csv(
    OUTPUT_DIR / "part_d_separation.csv",
    index=False,
    float_format="%.4f",
)

print("\nCLUSTER SEPARATION")
print(separation_df.to_string(index=False))


# ---------------------------------------------------------
# 6. Name the three clusters by mean delay rate
# ---------------------------------------------------------

cluster_delay_rates = (
    hub_profile.groupby("cluster_id")["delay_rate"]
    .mean()
    .sort_values()
)

cluster_to_segment = {
    cluster_id: segment
    for cluster_id, segment in zip(
        cluster_delay_rates.index,
        SEGMENT_NAMES,
    )
}

hub_profile["segment"] = hub_profile["cluster_id"].map(
    cluster_to_segment
)

hub_segments = hub_profile[
    ["hub_id", "segment", "hub_city", "hub_tier"]
].copy()

hub_segments = hub_segments.sort_values("hub_id").reset_index(drop=True)

hub_segments.to_csv(
    OUTPUT_DIR / "hub_segments.csv",
    index=False,
)


# ---------------------------------------------------------
# 7. Create segment-level summary
# ---------------------------------------------------------

segment_profile = (
    hub_profile.groupby("segment", as_index=False)
    .agg(
        hubs=("hub_id", "count"),
        mean_delivery_hours=("mean_delivery_hours", "mean"),
        delay_rate=("delay_rate", "mean"),
        mean_distance_km=("mean_distance_km", "mean"),
        mean_n_stops=("mean_n_stops", "mean"),
    )
)

segment_order = {
    "Standard": 0,
    "Watch": 1,
    "Escalate": 2,
}

segment_profile["_order"] = segment_profile["segment"].map(segment_order)
segment_profile = (
    segment_profile.sort_values("_order")
    .drop(columns="_order")
    .reset_index(drop=True)
)

segment_profile.to_csv(
    OUTPUT_DIR / "segment_profile.csv",
    index=False,
    float_format="%.4f",
)


# ---------------------------------------------------------
# 8. Identify mismatches between hub tier and segment
# ---------------------------------------------------------

hub_segments["expected_segment"] = (
    hub_segments["hub_tier"].map(TIER_TO_SEGMENT)
)

mismatches = hub_segments[
    hub_segments["segment"] != hub_segments["expected_segment"]
].copy()

mismatches.to_csv(
    OUTPUT_DIR / "tier_vs_segment_mismatch.csv",
    index=False,
)


# ---------------------------------------------------------
# 9. Print summary and output files
# ---------------------------------------------------------

print("\nHUB SEGMENTS")
print(hub_segments.to_string(index=False))

print("\nSEGMENT PROFILE")
print(segment_profile.round(4).to_string(index=False))

print("\nTIER / SEGMENT MISMATCHES")
print(mismatches.to_string(index=False))
print(f"\nMismatched hubs: {len(mismatches)}")

print("\nPART D SEGMENTATION COMPLETE")
print("Created:")
print(" - outputs/hub_profile.csv")
print(" - outputs/part_d_separation.csv")
print(" - outputs/hub_segments.csv")
print(" - outputs/segment_profile.csv")
print(" - outputs/tier_vs_segment_mismatch.csv")
