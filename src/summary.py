import pandas as pd
import numpy as np
# get_ipython().run_line_magic('pylab', 'inline')
data_cols = ['latitude', 'longitude',
             'height', 'days_float',
             'rec_date', 'rec_time',
             'timestamp', 'user_id',
             'trip_id', 'transit']
tpoints = (pd.read_csv('010_trip_labeled.csv',
                       index_col=0,
                       names=data_cols,
                       header=0)
             .drop('days_float' ,axis=1)
             .assign(timestamp = lambda x: pd.to_datetime(x.timestamp))
             .sort_values(['trip_id','timestamp'])
          )
tpoints = tpoints.loc[(~tpoints.trip_id.isnull())].copy()
tpoints.head(10)
(tpoints.groupby('trip_id').transit.nunique().max())
tpoints['time_delta'] = (tpoints.timestamp - \
                         tpoints.groupby(['trip_id']).timestamp.shift(1))
tpoints['dt_seconds'] = tpoints['time_delta'].dt.seconds
trip_dt = tpoints.groupby(['trip_id']).dt_seconds.max()
print(trip_dt.mean(), trip_dt.median())
trip_dt.quantile(np.arange(0,1,0.1))
rad_coord = np.radians(tpoints[['latitude','longitude']])
# same DF with the values shifted so that each set of coordinates represents the previous point
# in a given trip for a given index in rad_coord/tpoints
prev_rad_coord = np.radians(tpoints.groupby('trip_id').shift(1)[['latitude','longitude']])
# use haversine formula to compute distance in miles
lat1 = prev_rad_coord['latitude']
lon1 = prev_rad_coord['longitude']
lat2 = rad_coord['latitude']
lon2 = rad_coord['longitude']
dlon = lon2 - lon1
dlat = lat2 - lat1
a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
# 3965 == radius of the earth in miles - sub 6371 for km distance
delta_dist = 2 * np.arcsin(np.sqrt(a)) * 3956
tpoints['delta_dist'] = delta_dist
tpoints['speed'] = tpoints['delta_dist'] / (tpoints['dt_seconds'] / 3600. )
import time
quantum_start = time.time()
# for k in range(32):

tpoints.query('transit == "train"').groupby('trip_id').speed.max().describe()
trip_start = tpoints.groupby('trip_id').timestamp.transform('first')
trip_end = tpoints.groupby('trip_id').timestamp.transform('last')
trip_time = trip_end - trip_start
reg_walks = tpoints.groupby('trip_id').speed.transform('max') <= 15
min_time = ((trip_time.dt.seconds / 60.) >= 5) & ((trip_time.dt.seconds / 60.) <= 15)
match_walks = tpoints.loc[reg_walks & min_time,'trip_id'].unique()
ex_walk = tpoints.trip_id == match_walks[-1]
quantum_end = (time.time() - quantum_start)
print("Query time: {0}".format(quantum_end))
print(len(match_walks))
print('example trip:', match_walks[-1])
print('number of GPS observations:', len(tpoints.loc[ex_walk]))
adj_lat = tpoints.loc[ex_walk,'latitude'] - tpoints.loc[ex_walk,'latitude'].mean()
adj_lon = tpoints.loc[ex_walk,'longitude'] - tpoints.loc[ex_walk,'longitude'].mean()
data_plot = pd.concat([(adj_lat * 1000),(adj_lon * 1000)],axis=1).plot(x='longitude',y='latitude',marker='o')
from matplotlib import pyplot as plt
plt.boxplot((adj_lat *1000))
plt.show()
# print(pd)
