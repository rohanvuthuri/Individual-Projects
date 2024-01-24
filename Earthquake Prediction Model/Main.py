import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import datetime
import time
import tensorflow as tf
from keras import Sequential
from keras.layers import Dense

from mpl_toolkits.basemap import Basemap

from sklearn.model_selection import train_test_split

data = pd.read_csv("database.csv")
data.columns

data = data[["Date", "Time", "Latitude", "Longitude", "Depth", "Magnitude"]]
data.head()
#print(data)


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
#print(final_data)

'''x = final_data[["Timestamp", "Latitude", "Longitude"]]
y = final_data[["Magnitude", "Depth"]]

model = Sequential([
    Dense(units=25, activation='sigmoid'),
    Dense(units=15, activation='sigmoid'),
    Dense(units=1, activation='sigmoid'),
])
model.compile
print(model)'''

m = Basemap(projection='mill', llcrnrlat=-80, urcrnrlat=80, llcrnrlon=-180, urcrnrlon=180, lat_ts=20, resolution='c')
longitudes = data["Longitude"].tolist()
latitudes = data["Latitude"].tolist()
x,y = m(longitudes, latitudes)

fig = plt.figure(figsize = (12,10))
plt.title("Affected Areas")
m.plot(x, y, "o", markersize = 2, color = 'blue')
m.drawcoastlines()
m.fillcontinents(color='coral',lake_color='aqua')
m.drawmapboundary()
m.drawcountries()
#plt.show() 

x = final_data[["Timestamp", "Latitude", "Longitude"]]
y = final_data[["Magnitude", "Depth"]]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
#print(x_train.shape, x_test.shape, y_train.shape, x_test.shape)

model = Sequential([
    Dense(units=25, activation='relu'),
    Dense(units=15, activation='relu'),
    #Turns the data into a vector probability
    Dense(units=1, activation='softmax'),
])
'''model = Sequential()
model.add(Dense(units=25, activation='relu', input_shape=(3,)))
model.add(Dense(units=15, activation='relu'))
model.add(Dense(units=1, activation='softmax'))'''

model.compile(optimizer="SGD", loss = "squared_hinge", metrics=["accuracy"])
#fix this line of code
model.fit(x_train, y_train, batch_size=10, epochs=20, verbose=1, validation_data= (x_test, y_test))

[test_loss, test_acc] = model.evaluate(x_test, y_test)
print("Test loss is: " + test_loss + " and Test actual is: " + test_acc)