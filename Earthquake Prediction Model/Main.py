import csv
import datetime as dt
import os
import time
from pathlib import Path

import numpy as np


# Keep matplotlib cache inside the project so plotting works in restricted environments.
PROJECT_ROOT = Path(__file__).resolve().parent
MPL_CACHE_DIR = PROJECT_ROOT / ".matplotlib"
MPL_CACHE_DIR.mkdir(exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CACHE_DIR))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap


DATA_FILE = PROJECT_ROOT / "database.csv"
PLOT_FILE = PROJECT_ROOT / "predicted_earthquake_location.png"
TEST_RATIO = 0.2
RANDOM_SEED = 42
DEFAULT_NEIGHBORS = 12
EVAL_SAMPLE_SIZE = 250


def parse_timestamp(date_value, time_value):
    raw_value = f"{date_value} {time_value}".strip()
    formats = (
        "%m/%d/%Y %H:%M:%S",
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%dT%H:%M:%S.%fZ",
        "%Y-%m-%dT%H:%M:%S.%f",
        "%Y-%m-%dT%H:%M:%S",
    )

    for fmt in formats:
        try:
            parsed = dt.datetime.strptime(raw_value, fmt)
            return time.mktime(parsed.timetuple())
        except ValueError:
            continue

    return None


def load_dataset(csv_path):
    features = []
    targets = []

    with csv_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)

        for row in reader:
            timestamp = parse_timestamp(row["Date"], row["Time"])
            if timestamp is None:
                continue

            try:
                latitude = float(row["Latitude"])
                longitude = float(row["Longitude"])
                magnitude = float(row["Magnitude"])
                depth = float(row["Depth"])
            except (TypeError, ValueError):
                continue

            features.append([timestamp, latitude, longitude])
            targets.append([magnitude, depth])

    if not features:
        raise ValueError("No valid earthquake rows were loaded from database.csv.")

    return np.asarray(features, dtype=np.float64), np.asarray(targets, dtype=np.float64)


def split_dataset(features, targets, test_ratio=TEST_RATIO, seed=RANDOM_SEED):
    rng = np.random.default_rng(seed)
    indices = rng.permutation(len(features))
    split_index = int(len(features) * (1 - test_ratio))

    train_idx = indices[:split_index]
    test_idx = indices[split_index:]

    return (
        features[train_idx],
        features[test_idx],
        targets[train_idx],
        targets[test_idx],
    )


def normalize_features(train_features, other_features):
    mean = train_features.mean(axis=0)
    std = train_features.std(axis=0)
    std[std == 0] = 1.0

    train_scaled = (train_features - mean) / std
    other_scaled = (other_features - mean) / std

    return train_scaled, other_scaled, mean, std


def knn_predict(train_features, train_targets, query_features, neighbors=DEFAULT_NEIGHBORS):
    distances = np.linalg.norm(train_features - query_features, axis=1)
    nearest_indices = np.argsort(distances)[:neighbors]

    nearest_distances = distances[nearest_indices]
    if np.any(nearest_distances == 0):
        zero_distance_index = nearest_indices[np.argmin(nearest_distances)]
        return train_targets[zero_distance_index], zero_distance_index

    weights = 1.0 / (nearest_distances + 1e-9)
    weighted_targets = train_targets[nearest_indices] * weights[:, None]
    prediction = weighted_targets.sum(axis=0) / weights.sum()

    return prediction, nearest_indices[0]


def evaluate_model(train_features, train_targets, test_features, test_targets, neighbors=DEFAULT_NEIGHBORS):
    sample_size = min(EVAL_SAMPLE_SIZE, len(test_features))
    sampled_features = test_features[:sample_size]
    sampled_targets = test_targets[:sample_size]

    predictions = np.array(
        [knn_predict(train_features, train_targets, feature, neighbors)[0] for feature in sampled_features]
    )

    absolute_error = np.abs(predictions - sampled_targets)
    squared_error = (predictions - sampled_targets) ** 2

    mae = absolute_error.mean(axis=0)
    rmse = np.sqrt(squared_error.mean(axis=0))

    return sample_size, mae, rmse


def prompt_float(prompt_text, min_value, max_value):
    while True:
        raw_value = input(prompt_text).strip()

        try:
            value = float(raw_value)
        except ValueError:
            print("Please enter a valid number.")
            continue

        if value < min_value or value > max_value:
            print(f"Value must be between {min_value} and {max_value}.")
            continue

        return value


def save_prediction_map(user_latitude, user_longitude, nearest_latitude, nearest_longitude, predicted_target):
    figure = plt.figure(figsize=(12, 6))
    basemap = Basemap(
        projection="mill",
        llcrnrlat=-80,
        urcrnrlat=80,
        llcrnrlon=-180,
        urcrnrlon=180,
        lat_ts=20,
        resolution="c",
    )

    basemap.drawcoastlines(linewidth=0.5)
    basemap.drawcountries(linewidth=0.5)
    basemap.drawmapboundary(fill_color="#d9eff7")
    basemap.fillcontinents(color="#f4e7c5", lake_color="#d9eff7")

    user_x, user_y = basemap(user_longitude, user_latitude)
    nearest_x, nearest_y = basemap(nearest_longitude, nearest_latitude)

    basemap.plot(user_x, user_y, "bo", markersize=7, label="Input location")
    basemap.plot(nearest_x, nearest_y, "ro", markersize=6, label="Closest historical event")

    plt.legend(loc="lower left")
    plt.title(
        "Predicted Earthquake Conditions\n"
        f"Magnitude {predicted_target[0]:.2f}, Depth {predicted_target[1]:.2f} km"
    )
    plt.tight_layout()
    plt.savefig(PLOT_FILE, dpi=150)
    plt.close(figure)


def main():
    features, targets = load_dataset(DATA_FILE)
    train_x, test_x, train_y, test_y = split_dataset(features, targets)
    train_x_scaled, test_x_scaled, feature_mean, feature_std = normalize_features(train_x, test_x)

    print(f"Loaded {len(features):,} earthquake records from {DATA_FILE.name}.")
    print("Prediction model: weighted k-nearest neighbors using timestamp, latitude, and longitude.")

    latitude = prompt_float("Enter latitude (-90 to 90): ", -90.0, 90.0)
    longitude = prompt_float("Enter longitude (-180 to 180): ", -180.0, 180.0)

    current_timestamp = time.mktime(dt.datetime.now().timetuple())
    query = np.asarray([[current_timestamp, latitude, longitude]], dtype=np.float64)
    query_scaled = (query - feature_mean) / feature_std

    prediction, nearest_index = knn_predict(train_x_scaled, train_y, query_scaled[0])
    nearest_event = train_x[nearest_index]

    sample_size, mae, rmse = evaluate_model(train_x_scaled, train_y, test_x_scaled, test_y)

    print(f"Predicted magnitude: {prediction[0]:.2f}")
    print(f"Predicted depth: {prediction[1]:.2f} km")
    print(f"Closest historical event: lat {nearest_event[1]:.3f}, lon {nearest_event[2]:.3f}")
    print(
        "Evaluation on held-out data sample "
        f"({sample_size} rows): magnitude MAE {mae[0]:.2f}, depth MAE {mae[1]:.2f} km"
    )
    print(
        "Evaluation on held-out data sample "
        f"({sample_size} rows): magnitude RMSE {rmse[0]:.2f}, depth RMSE {rmse[1]:.2f} km"
    )

    save_prediction_map(
        user_latitude=latitude,
        user_longitude=longitude,
        nearest_latitude=nearest_event[1],
        nearest_longitude=nearest_event[2],
        predicted_target=prediction,
    )
    print(f"Saved map to {PLOT_FILE.name}")


if __name__ == "__main__":
    main()
