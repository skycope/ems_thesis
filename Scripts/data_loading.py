# this script loads the necessary data
# libraries
import pandas as pd
import geopandas
import os
import numpy as np



# temporal distribution of demand
# Get the absolute path of the current script
script_path = os.path.abspath(__file__)

# Get the parent directory of the current script
script_dir = os.path.dirname(script_path)

# Set the working directory to the sibling directory 'input_data'
input_data_dir = os.path.join(script_dir, '..', 'input_data')
os.chdir(input_data_dir)

temporal_dist = pd.read_csv('weekly_calls.csv')

# read in ft_rates_output
ft_rates_output = pd.read_csv('ft_rates_output.csv')

# ward-to-facility-probability distributions
ppt_wards_facilities = pd.read_csv('ppt_non_p1.csv')
# in the columns, remove to_ward_ prefix
ppt_wards_facilities.columns = [col.replace('to_ward_', '') for col in ppt_wards_facilities.columns]
# format all columns as numeric
ppt_wards_facilities = ppt_wards_facilities.apply(pd.to_numeric, errors='coerce')
p2_wards_facilities = pd.read_csv('primary_non_p1.csv')
# and again here
p2_wards_facilities.columns = [col.replace('to_ward_', '') for col in p2_wards_facilities.columns]
# format all columns as numeric
p2_wards_facilities = p2_wards_facilities.apply(pd.to_numeric, errors='coerce')
p1_wards_facilities = pd.read_csv('primary_p1.csv')
# format all columns as numeric
p1_wards_facilities = p1_wards_facilities.apply(pd.to_numeric, errors='coerce')
# and again here
p1_wards_facilities.columns = [col.replace('to_ward_', '') for col in p1_wards_facilities.columns]



# inter-ward travel times from google maps
gmaps_travel = pd.read_csv('gmaps_travel.csv').fillna(0)

# ward-level spatial demand distribuition, by calltype
p1_dist = pd.read_csv('primary_p1_dist.csv')
p2_dist = pd.read_csv('primary_p2_dist.csv')
ppt_dist = pd.read_csv('ppt_p2_dist.csv')

# merge p1_dist, ppt_p2_dist, p2_dist into one dataframe
# this is the spatial demand distribution for each ward
combined = pd.merge(p1_dist, ppt_dist, on='from_ward')
combined = pd.merge(combined, p2_dist, on='from_ward')

# coordinates of wards with facilities
facilities_dist = pd.read_csv('facilities.csv')

# read in nmb_wards.shx
wards_shp = geopandas.read_file('nmb_wards.shx')
# add ward names to wards_shp, using row index as ward_id
wards_shp['ward_id'] = wards_shp.index
# ward centroids
# get ward centroid coordinates and store them in a df called wards_centroids
wards_centroids = wards_shp.centroid
# convert ward_centroids from shp points to a df with lat and lon columns
wards_centroids = wards_centroids.apply(lambda x: pd.Series([x.x, x.y]))
# set columns to lat and lon, and add ward_id from 1:60
wards_centroids.columns = ['lon', 'lat']
wards_centroids['ward_id'] = wards_centroids.index + 1



# google maps travel times
df = pd.read_csv('gmaps_tts.csv')

# initialise blank data frame
dists_to_central_wards = pd.DataFrame({
    'ward_id': np.arange(1, 61),
    'ward_25': np.nan,
    'ward_5': np.nan,
    'ward_51': np.nan,
    'ward_58': np.nan,
    'ward_34': np.nan
})
# now we need to add the travel times to each of the central wards
# do a loop
for i in range(1, 61):
    dists_to_central_wards.loc[dists_to_central_wards.ward_id == i, 'ward_25'] = df[(df.from_ward == i) & (df.to_ward == 25)].travel_time.values[0]/600
    dists_to_central_wards.loc[dists_to_central_wards.ward_id == i, 'ward_5'] = df[(df.from_ward == i) & (df.to_ward == 5)].travel_time.values[0]/600
    dists_to_central_wards.loc[dists_to_central_wards.ward_id == i, 'ward_51'] = df[(df.from_ward == i) & (df.to_ward == 51)].travel_time.values[0]/600
    dists_to_central_wards.loc[dists_to_central_wards.ward_id == i, 'ward_58'] = df[(df.from_ward == i) & (df.to_ward == 58)].travel_time.values[0]/600
    dists_to_central_wards.loc[dists_to_central_wards.ward_id == i, 'ward_34'] = df[(df.from_ward == i) & (df.to_ward == 34)].travel_time.values[0]/600


# now get the argmin gor each row and add it to the df as a column
dists_to_central_wards['closest_central_ward'] = dists_to_central_wards[['ward_25', 'ward_5', 'ward_51', 'ward_58', 'ward_34']].idxmin(axis=1)
# now remove the ward_ prefix from the column
dists_to_central_wards['closest_central_ward'] = dists_to_central_wards['closest_central_ward'].str.replace('ward_', '')
# now convert to numeric
# dists_to_central_wards['closest_central_ward'] = pd.to_numeric(dists_to_central_wards['closest_central_ward'])
# 

# set from_ward and to_ward to numeric
df['from_ward'] = pd.to_numeric(df['from_ward'])
df['to_ward'] = pd.to_numeric(df['to_ward'])
df['duration'] = df['travel_time']/60
# filter df for from_ward == 39 and print
# make a new data frame that has these columns: ward_id, time to ward 5, 51, 58, and 34
# make it show the time to ward 5, 51, 58, and 34 for each ward

# make heatmap
gmaps_travel_time_matrix = df.pivot_table(index='from_ward', columns='to_ward', values='duration')
# apply argmin to each row, while filtering the columns to be the central wards: 5, 51, 58, 34
# Make the distance matrix symmetric by mirroring the lower triangle
gmaps_travel_time_matrix_symmetric = gmaps_travel_time_matrix + gmaps_travel_time_matrix.T - np.diag(np.diag(gmaps_travel_time_matrix))


# print a filtered version which only
# ## more data inputs
# set up dispatch and demand node coordinates
# add dispatch node coordinates
dispatch_pts_outline = pd.DataFrame({
    'x': facilities_dist.x,
    'y': facilities_dist.y,
    'ward_id': facilities_dist.to_ward
})

# get call rates per tick for specified number of days, at 10-minute resolution
n_ticks = 6*24*14 # 14 days


# we first get the overall call rates in the whole region for each call type
overall_p1_rates = np.array(np.tile(temporal_dist[temporal_dist.calltype == 'p1'].call_rate, 2))
overall_p2_rates = np.array(np.tile(temporal_dist[temporal_dist.calltype == 'p2'].call_rate, 2))
overall_ppt_rates = np.array(np.tile(temporal_dist[temporal_dist.calltype == 'ppt'].call_rate, 2))


# we then get the call rates for each ward, by multiplying the overall call rates by 
# the proportion of calls in each ward
rates_p1 = np.tile(p1_dist.prop, (n_ticks, 1)).T*overall_p1_rates
rates_p2 = np.tile(p2_dist.prop, (n_ticks, 1)).T*overall_p2_rates
rates_ppt = np.tile(ppt_dist.prop, (n_ticks, 1)).T*overall_ppt_rates

# # Create a graph from the distance matrix, where you omit the first column
# # print dimensions of np.array(gmaps_travel_time_matrix)[1:, 1:]
# # Create a mapping dictionary for node indices to location IDs
# column_names = list(gmaps_travel_time_matrix.columns)
# index_names = list(gmaps_travel_time_matrix.index)
# label_mapping = {i: index_names[i] for i in range(len(index_names))}

# G = nx.from_numpy_array(np.array(gmaps_travel_time_matrix_symmetric))

# # Compute the Kamada-Kawai layout
# pos = nx.kamada_kawai_layout(G, dist=gmaps_travel_time_matrix_symmetric)
# # add 1 to the node labels
# # Draw the graph using the computed layout
# # Draw the graph using the computed layout and the label mapping

# # Find the indices of the desired nodes
# highlighted_nodes = [index_names.index(loc_id) for loc_id in [5, 51, 58, 34]]

# # Hardcoded lists of nodes for each closest central ward
# closest_to_5 = [1, 2, 3, 4, 6, 7, 8, 9, 10, 11, 12, 14, 15, 16, 17, 18, 25, 39]
# closest_to_34 = [13, 20, 26, 27, 28, 29, 30, 31, 32, 33, 35, 36, 37, 38, 40]
# closest_to_51 = [41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 52]
# closest_to_58 = [19, 21, 22, 23, 24, 53, 54, 55, 56, 57, 59, 60]
# central_nodes = [5, 51, 58, 34]

# # Apply the MDS layout
# embedding = MDS(n_components=2, dissimilarity='precomputed', random_state=42)
# pos_array = embedding.fit_transform(gmaps_travel_time_matrix_symmetric)
# pos = {i: pos_array[i] for i in range(len(pos_array))}

# # Draw the graph using the MDS layout and the label mapping
# nx.draw(G, pos, labels=label_mapping, with_labels=True, node_size=200, node_color='skyblue', font_size=8)
# # Highlight the nodes with different colors based on the closest central ward
# nx.draw_networkx_nodes(G, pos, nodelist=[node - 1 for node in closest_to_5], node_color='r', node_size=200)
# nx.draw_networkx_nodes(G, pos, nodelist=[node - 1 for node in closest_to_34], node_color='g', node_size=200)
# nx.draw_networkx_nodes(G, pos, nodelist=[node - 1 for node in closest_to_51], node_color='b', node_size=200)
# nx.draw_networkx_nodes(G, pos, nodelist=[node - 1 for node in closest_to_58], node_color='m', node_size=200)

# plt.show()

# plt.show()
