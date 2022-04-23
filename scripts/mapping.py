#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import folium
import glob as glob
import src.preprocess as pre
import numpy as np
import io
from PIL import Image
import math as m
import time
# GEOLIFE TESTS
gpsHeader = ['Latitude', 'Longitude', 'Zero', 'Altitude', 'Num of Days', 'Date', 'Time']

# MDC TESTS
# gpsHeader = ['Index', 'UID', 'Date', 'Time', 'Longitude', 'Latitude']


# In[2]:


# ONE FILE
pathToFile = "C:\\Users\\Betis\\PycharmProjects\\Big_Data\\src\\user_by_month\\000\\2008_12.csv"
df = pd.read_csv(pathToFile, names=gpsHeader)

# In[3]:


# WHOLE DIRECTORY

pathToDir = 'C:\\Users\\Betis\\PycharmProjects\\Big_Data\\src\\user_by_month\\000\\'
glob = glob.glob(pathToDir + '*')

df = pd.concat([pd.read_csv(f, names=gpsHeader) for f in glob])  # , header=gpsHeader)

# In[4]:


## Bounding Box to save time
bb = pre.fetchGeoLocation('Beijing, China')
df = pre.dropOutlyingData(df, bb)

print(df.head())

# In[5]:


# time = df[['Time']].to_numpy()
df = df[['Latitude', 'Longitude']]  # change to your column names, assume the columns are sorted by time
points = [tuple(x) for x in df.to_numpy()]

# for i in range(0, len(points)):
#     x = points[i]
#     a = float(f'{float(f"{x[0]:.4g}"):g}')
#     b = float(f'{float(f"{x[1]:.10g}"):g}')
#     points[i] = (a, b)

ave_lat = sum(p[0] for p in points) / len(points)
ave_lon = sum(p[1] for p in points) / len(points)

# Load map centred on average coordinates
my_map = folium.Map(location=[ave_lat, ave_lon], zoom_start=14)

# In[6]:


tally = {}

for key in points:
    # Check for duplicates
    count = points.count(key)
    if (count > 1):
        # Add 1 to existing key, otherwise set to 1
        tally[key] = tally.setdefault(key, 0) + 1

print(len(tally))
uniqueTuples = np.unique(points, axis=0)
print(len(uniqueTuples))

mostFreqLocation = max(tally, key=tally.get)
print('Unique Tally Values')
print(np.unique(list(tally.values())))
print('Most Frequent Location')
print(mostFreqLocation)

# In[7]:


maxTally = max(tally.values())
minTally = min(tally.values())

for key in tally.keys():
    folium.CircleMarker(key, radius=m.log(tally[key], maxTally), color='red').add_to(my_map)

folium.Marker(mostFreqLocation).add_to(my_map)

# In[8]:


my_map.save("./poi.html")

# In[ ]:


tally = {}
# add a markers
for i in range(0, len(points)):
    folium.CircleMarker(points[i], radius=0.5, color='red').add_to(my_map)
    for j in range(0, len(points)):
        if (i != j and points[i] == points[j]):
            key = points[i]
            tally[key] = tally.setdefault(key, 0) + 1

for key in tally.keys():
    folium.Marker(key).add_to(my_map)
# add lines
# folium.PolyLine(points, color="red", weight=2.5, opacity=1).add_to(my_map)
print(tally)
# Save map
my_map.save("./out.html")

# In[ ]:


mostFreqLocation = max(tally, key=tally.get)
print(tally.values())
print(mostFreqLocation)

# In[ ]:


print(len(tally))

# In[ ]:


print((time))

