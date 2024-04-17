from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import datetime
import time
from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt
import folium
from keras.layers import Input, Dense

data = pd.read_csv("/Users/Rohan/Documents/Individual-Projects/Earthquake Prediction Model/database.csv")
data.columns

data = data[["Date", "Time", "Latitude", "Longitude", "Depth", "Magnitude"]]
data.head()
timestamp = []
for d, t in zip(data["Date"], data["Time"]):
    try:
        ts = datetime.datetime.strptime(d+' '+t, '%m/%d/%Y %H:%M:%S')
        timestamp.append(time.mktime(ts.timetuple()))
    except ValueError:
        # print('ValueError')
        timestamp.append("ValueError")
timeStamp = pd.Series(timestamp)
data["Timestamp"] = timeStamp.values
final_data = data.drop(["Date","Time"], axis=1)
final_data = final_data[final_data.Timestamp != "ValueError"]
final_data.head()
x = final_data[["Timestamp", "Latitude", "Longitude"]]
y = final_data[["Magnitude", "Depth"]]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

model = Sequential()
input_shape = Input(shape=(x_train.shape[1], ))
model.add(input_shape)
model.add(Dense(25, activation='relu'))
model.add(Dense(15, activation='relu'))
model.add(Dense(2, activation='linear'))  # Output layer for predicting latitude and longitude
model.compile(optimizer="adam", loss="mse", metrics=['accuracy'])

model.fit(x_train, y_train, batch_size=10, epochs=20, verbose=1, validation_data=(x_test, y_test))


current_time = datetime.datetime.now()
timestamp_value = int(time.mktime(current_time.timetuple()))
latitude_guess = float(input("Enter latitude: "))
longitude_guess = float(input("Enter longitude: "))
new_data = np.array([[timestamp_value, latitude_guess, longitude_guess]])
predicted_coordinates = model.predict(new_data)
print("Predicted Coordinates: Longitude = {:.4f}, Latitude = {:.4f}".format(predicted_coordinates[0][0], predicted_coordinates[0][1]))

evaluation = model.evaluate(x_test, y_test)
print(f"Loss: {evaluation[0]}, Accuracy: {evaluation[1]}")

# Setting up the map
m = Basemap(projection='mill', llcrnrlat=-80, urcrnrlat=80, llcrnrlon=-180, urcrnrlon=180, lat_ts=20, resolution='c')
m.drawcoastlines()
m.drawcountries()

# Plotting predicted coordinates
x, y = m(predicted_coordinates[0][1], predicted_coordinates[0][0])
m.plot(x, y, 'bo', markersize=10000)  # 'bo' for blue dot
#Blue dot will show where the next nearest location of earthquake
plt.title('Predicted Earthquake Location')
plt.show()
