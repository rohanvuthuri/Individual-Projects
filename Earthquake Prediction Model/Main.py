import datetime
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from keras.callbacks import EarlyStopping
from keras.layers import Dense, Input
from keras.models import Sequential
from keras.utils import set_random_seed
from mpl_toolkits.basemap import Basemap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Neural networks start from random weights, so two runs can give slightly
# different results. Fixing the seed makes every run identical, which lets us
# compare "before vs after" a change fairly.
set_random_seed(42)

# ---------------------------------------------------------------------------
# 1. Load the data
# ---------------------------------------------------------------------------
load_dotenv()
data = pd.read_csv(os.getenv("CSV_PATH"))
data = data[["Date", "Time", "Latitude", "Longitude", "Depth", "Magnitude"]]

# Turn "01/02/1965" + "13:44:18" into one number (seconds since 1970) so the
# network can use it. A few rows use a different date format; we mark those
# as NaN (missing) instead of the string "ValueError" so dropna() removes them
# and every column stays numeric.
timestamps = []
for d, t in zip(data["Date"], data["Time"]):
    try:
        ts = datetime.datetime.strptime(d + " " + t, "%m/%d/%Y %H:%M:%S")
        timestamps.append(time.mktime(ts.timetuple()))
    except ValueError:
        timestamps.append(np.nan)
data["Timestamp"] = timestamps
data = data.drop(["Date", "Time"], axis=1).dropna()

# Inputs (what we know) and targets (what we want to predict).
x = data[["Timestamp", "Latitude", "Longitude"]]
y = data[["Magnitude", "Depth"]]

# Hold back 20% of the earthquakes as a test set. The network never trains on
# these, so they tell us how well it does on quakes it has never seen.
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# ---------------------------------------------------------------------------
# 2. Scale the inputs AND the targets  (the main fix)
# ---------------------------------------------------------------------------
# Why: the timestamp is in the billions while latitude/longitude are at most
# 180. Fed in raw, the timestamp blew every first-layer neuron up to exactly
# 0 or 1, so the network ignored location and could only learn one answer.
#
# StandardScaler rescales each column to "how many standard deviations from
# the average", so every value lands roughly between -3 and 3 and each input
# gets an equal say.
#
# We fit the scaler on the TRAINING data only. If it also looked at the test
# set, information about the test quakes would leak into training and our
# test score would look better than it really is.
x_scaler = StandardScaler()
x_train_scaled = x_scaler.fit_transform(x_train)
x_test_scaled = x_scaler.transform(x_test)

# Targets get scaled too. Depth varies by ~120 km but magnitude only by ~0.4,
# so with raw values the loss was ~99.99% depth error and the network had no
# reason to care about magnitude. Scaled, both targets count equally.
# (We undo this scaling later to get answers back in real units.)
y_scaler = StandardScaler()
y_train_scaled = y_scaler.fit_transform(y_train)

# ---------------------------------------------------------------------------
# 3. Build the network
# ---------------------------------------------------------------------------
# Same shape as before: 3 inputs -> 25 -> 15 -> 2 outputs.
#
# Changed sigmoid -> ReLU in the hidden layers. Sigmoid flattens out near 0
# and 1, and where it is flat the slope is ~0, so the weights stop updating
# ("vanishing gradients"). ReLU (max(0, x)) keeps a slope of 1 for any
# positive value, so the network keeps learning.
#
# The output layer stays linear because magnitude and depth are any real
# number, not a 0-1 probability.
model = Sequential([
    Input(shape=(x_train_scaled.shape[1],)),
    Dense(25, activation="relu"),
    Dense(15, activation="relu"),
    Dense(2, activation="linear"),  # [magnitude, depth] (in scaled units)
])

# Loss = MSE (mean squared error): punishes big misses hard, which is what we
# want the optimizer to minimize.
# Metric = MAE (mean absolute error): "on average, how far off are we?".
# We dropped 'accuracy' because it only makes sense for picking categories.
# Here it was really checking "is depth bigger than magnitude?", which is
# true 98% of the time no matter what the network learns.
model.compile(optimizer="adam", loss="mse", metrics=["mae"])

# ---------------------------------------------------------------------------
# 4. Train
# ---------------------------------------------------------------------------
# Early stopping: watch the validation loss and stop once it hasn't improved
# for 5 epochs in a row, then roll back to the best weights seen. Before, the
# loss stopped improving around epoch 13 but training kept going anyway.
#
# validation_split=0.1 sets aside 10% of the TRAINING data to watch. We don't
# use the test set for this: early stopping picks the epoch that scores best
# on whatever it watches, so watching the test set would quietly tune the
# model to it and make the final test score look better than it really is.
#
# batch_size=32 (was 10): bigger batches mean fewer weight updates per epoch,
# so training runs about 3x faster with no real loss in quality here.
early_stop = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)
model.fit(
    x_train_scaled,
    y_train_scaled,
    batch_size=32,
    epochs=100,  # an upper limit; early stopping usually ends well before this
    validation_split=0.1,
    callbacks=[early_stop],
    verbose=2,  # one line per epoch instead of a progress bar
)

# ---------------------------------------------------------------------------
# 5. Evaluate on the held-out test set, in real units
# ---------------------------------------------------------------------------
# The network outputs scaled values, so inverse_transform converts them back
# to real magnitude and km before we measure the error.
test_pred = y_scaler.inverse_transform(model.predict(x_test_scaled, verbose=0))
nn_mae = np.abs(test_pred - y_test.to_numpy()).mean(axis=0)

# A number like "off by 40 km" means nothing on its own, so compare it with the
# laziest possible model: always guess the training average. The old network
# was doing exactly this. If we can't beat it, the network learned nothing.
baseline_mae = np.abs(y_train.mean().to_numpy() - y_test.to_numpy()).mean(axis=0)

print(f"\nTest set ({len(y_test)} earthquakes the network never saw):")
print(f"  Network:         magnitude off by {nn_mae[0]:.3f}, depth off by {nn_mae[1]:.1f} km")
print(f"  Always-average:  magnitude off by {baseline_mae[0]:.3f}, depth off by {baseline_mae[1]:.1f} km")

# ---------------------------------------------------------------------------
# 6. Predict for a location you choose
# ---------------------------------------------------------------------------
current_time = datetime.datetime.now()
timestamp_value = time.mktime(current_time.timetuple())
latitude_guess = float(input("\nEnter latitude: "))
longitude_guess = float(input("Enter longitude: "))

# The input goes through the same steps as the training data: scale it with
# the training scaler, predict, then unscale the output. A DataFrame with the
# same column names keeps scikit-learn happy.
new_data = pd.DataFrame([[timestamp_value, latitude_guess, longitude_guess]], columns=x.columns)
prediction = y_scaler.inverse_transform(model.predict(x_scaler.transform(new_data), verbose=0))
predicted_magnitude, predicted_depth = prediction[0]

# These are magnitude and depth, not coordinates (the old labels were wrong).
print(f"Predicted magnitude: {predicted_magnitude:.2f}")
print(f"Predicted depth: {predicted_depth:.1f} km")

# ---------------------------------------------------------------------------
# 7. Map
# ---------------------------------------------------------------------------
# Plot the location you entered and show the prediction in the title. Before,
# the code plotted (magnitude, depth) as if they were (longitude, latitude),
# which is why the dot always landed in the Arctic Ocean.
m = Basemap(projection="mill", llcrnrlat=-80, urcrnrlat=80, llcrnrlon=-180, urcrnrlon=180, lat_ts=20, resolution="c")
m.drawcoastlines()
m.drawcountries()

map_x, map_y = m(longitude_guess, latitude_guess)  # Basemap takes (lon, lat)
m.plot(map_x, map_y, "ro", markersize=6)
plt.title(f"Predicted earthquake at ({latitude_guess}, {longitude_guess})\n"
          f"Magnitude {predicted_magnitude:.2f}, depth {predicted_depth:.1f} km")
plt.show()
