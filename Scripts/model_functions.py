# this script contains the functions used by the model
# import libraries
import pandas as pd
import geopandas
import numpy as np
from shapely.geometry import Point
import json
from tqdm.auto import tqdm
import matplotlib.pyplot as plt

# get input data from data_loading
from data_loading import (wards_shp, gmaps_travel_time_matrix, 
                        p1_dist, p2_dist, ppt_dist, 
                        ppt_wards_facilities, p2_wards_facilities, p1_wards_facilities,
                        rates_ppt, rates_p2, rates_p1,
                        wards_centroids, ft_rates_output, dists_to_central_wards)

def get_area(ward_id):
    # return the closest central ward id
    return dists_to_central_wards.loc[dists_to_central_wards.ward_id==ward_id, 'closest_central_ward'].values[0]


# a vectorised version for multiple lats and longs.
# given shapefile of wards, and two lists of lats and longs, return a list of the nearest wards
def closest_ward(point, centroids):
    distances = centroids.distance(point)
    # get row number of closest centroid
    closest_centroid = distances.idxmin() + 1
    # return row number of closest centroid
    return closest_centroid 


def get_ward(wards_shp, lats, longs):
    points = geopandas.points_from_xy(longs, lats)
    # create a geodataframe from the point
    points_gfd = geopandas.GeoDataFrame(geometry=points)
    # get centroids of all wards
    centroids = wards_shp.centroid
    closest_centroids = points_gfd.apply(lambda row: closest_ward(row.geometry, centroids), axis=1)
    return closest_centroids.tolist()


# all remaining functions
# draw travel time from gamma distribution
# as a function of mean (distance) and standard deviation (sd)
def travel_time(from_ward, to_ward, travel_time_matrix, scaling, sd=0.01):
    sd = sd*scaling # scale sd as we scale the mean
    mean = travel_time_matrix.iloc[from_ward-1][to_ward]*scaling/10

    if mean == 0:
        mean = 0.5 # if the from_ward and to_ward are the same, set mean to 5 minutes
    # calculate variance from sd
    variance = sd ** 2
    # set shape parameter
    k = mean ** 2 / variance
    # set scale parameter
    theta = variance / mean
    # draw from gamma distribution
    return np.random.gamma(shape=k, scale=theta)

    
# function to return the sampled
# number of calls at each time step, demand node and priority
def sample_calls(ticks, rates_p1, rates_p2, rates_ppt):
    # number of wards
    n_demand_nodes = np.shape(rates_p1)[0]
    
    # initialise array to store sampled calls
    sampled_calls = np.zeros((3, n_demand_nodes, ticks), dtype='int')

    # populate the empty sampled_calls array with sampled calls
    # iterate through each time step
    # theta = 6.7 from GAM
    theta = 6.7
    for tick in range(ticks):
        # iterate through each demand node
        # for primary p1 calls
        for ward in range(len(rates_p1)):
            # in order to draw from a poisson distribution, we need to specify the mean
            # number of calls
            mean = rates_p1[ward, tick]
            n = theta
            p = theta/(theta + mean)
            sampled_calls[0][ward][tick] = np.random.negative_binomial(n=n, p=p, size=1)
        # for primary p2 calls
        for ward in range(len(rates_p2)):
            mean = rates_p2[ward, tick]
            n = theta
            p = theta/(theta + mean)
            sampled_calls[1][ward][tick] = np.random.negative_binomial(n=n, p=p, size=1)
        # for ppt p2 calls
        for ward in range(len(rates_ppt)):
            mean = rates_ppt[ward, tick]
            n = theta
            p = theta/(theta + mean)
            sampled_calls[2][ward][tick] = np.random.negative_binomial(n=n, p=p, size=1)
    return sampled_calls

# function that accepts the incident ward id and returns the destination ward by sampling
def sample_facilities(incident_ward, calltype):
    if calltype == 'ppt':
        probs = ppt_wards_facilities.loc[ppt_wards_facilities.from_ward == incident_ward, :].values[0][1:].tolist()
        # ensure they sum to 1 by dividing by the sum
        if sum(probs) == 0:
            probs = np.array([1/60]*60)
        else:
            probs = [prob/sum(probs) for prob in probs]
    elif calltype == 'p2':
        probs = p2_wards_facilities.loc[p2_wards_facilities.from_ward == incident_ward, :].values[0][1:].tolist()
        # ensure they sum to 1 by dividing by the sum
        if sum(probs) == 0:
            probs = np.array([1/60]*60)
        else:
            probs = [prob/sum(probs) for prob in probs]
    else:
        probs = p1_wards_facilities.loc[p1_wards_facilities.from_ward == incident_ward, :].values[0][1:].tolist()
        if sum(probs) == 0:
            probs = np.array([1/60]*60)
        else:
            probs = [prob/sum(probs) for prob in probs]
    
    # weighted sampling
    sampled_wards = np.random.choice(a=np.array(range(1, 61)), size=1, p=probs)
    # print('sampled ward is {} for incident ward {} and calltype {}'.format(sampled_wards, incident_ward, calltype))
    return sampled_wards

# function that, given demand point id and dispatch point ids, returns the closest dispatch point
def closest_dispatch(demand_point_id, dispatch_point_ids, travel_time_matrix):
    # filter travel_time_matrix so that from_ward == demand_point_id and column number is in dispatch_point_ids
    # add 1 to dispatch_point_ids to account for 0 indexing
    travel_time_matrix_filtered = travel_time_matrix.iloc[demand_point_id-1][dispatch_point_ids]
    # get the index of the minimum value
    min_index = travel_time_matrix_filtered.idxmin()
    return min_index


def available_vehicles(demand_type, vehicles, staff, prob_single_staff, prob_amb, prob_ip, reroute_p1=True):
    # extract prob_p1_single from prob_single_staff dictionary
    if demand_type == ['p1']:
        prob_single = prob_single_staff['p1']
        prob_amb = prob_amb['p1']
        prob_ip = prob_ip['p1']
    elif demand_type == ['p2']:
        prob_single = prob_single_staff['p2']
        prob_amb = prob_amb['p2']
        prob_ip = prob_ip['p2']
    else:
        prob_single = prob_single_staff['ppt']
        prob_amb = prob_amb['ppt']
        prob_ip = prob_ip['ppt']
    # generate random number for probability
    if np.random.uniform(0, 1) < prob_single:
        allow_single = True
    else:
        allow_single = False
    if np.random.uniform(0, 1) < prob_amb:
        require_amb = True
    else:
        require_amb = False
    if np.random.uniform(0, 1) < prob_ip:
        require_ip = True
    else:
        require_ip = False
    
    if not reroute_p1 and demand_type == ['p1']:
        demand_type = ['p2']
    
    # count number of idle staff of each type at each base
    # filter staff df to include only staff that are idle
    staff_count = staff.loc[staff['status'] == 'idle'].groupby(['base', 'type']).count().reset_index()[['base', 'type', 'id']]
    # rename columns
    staff_count.columns = ['base', 'type', 'n_idle']
    
    # if number of rows of staff_count > 0
    if staff_count.shape[0] > 0:
        # store ward IDs that have at least two IPs that are idle
        atleast_2_ip = staff_count.loc[(staff_count['type'] == 'ip') & (staff_count['n_idle'] >= 2)]['base'].values
        # store ward IDs that have at least two SPs that are idle
        atleast_2_sp = staff_count.loc[(staff_count['type'] == 'sp') & (staff_count['n_idle'] >= 2)]['base'].values
        # store ward ids with at least 1 sp
        atleast_1_sp = staff_count.loc[(staff_count['type'] == 'sp') & (staff_count['n_idle'] >= 1)]['base'].values
        # store wards with at least 1 ip
        atleast_1_ip = staff_count.loc[(staff_count['type'] == 'ip') & (staff_count['n_idle'] >= 1)]['base'].values

        if allow_single and require_ip:
            staff_wards = atleast_1_ip
        elif allow_single and not require_ip:
            staff_wards = np.concatenate((atleast_1_sp, atleast_1_ip))
        elif not allow_single and require_ip:
            staff_wards = np.concatenate((atleast_2_ip, np.intersect1d(atleast_1_sp, atleast_1_ip)))
        else:
            staff_wards = np.concatenate((atleast_2_ip, atleast_2_sp, np.intersect1d(atleast_1_sp, atleast_1_ip)))
        # remove duplicates
        staff_wards = np.unique(staff_wards)

        # now vehicles
        # if call is ppt, the vehicle doesn't need to be an ambulance
        if not require_amb:
            # get wards of vehicles that are idle at base
            idle_vehicle_wards = vehicles.loc[(vehicles['status'] == 'idle') & (vehicles['current_ward'] == vehicles['base_ward'])]['base_ward'].values
            # find the wards where there are enough staff and vehicles that are idle
            available_wards = np.intersect1d(staff_wards, idle_vehicle_wards)
            # get IDs of vehicles where status is idle, current ward is base ward and current ward is in available_wards
            idle_at_base_ids = vehicles.loc[(vehicles['status'] == 'idle') & (vehicles['current_ward'] == vehicles['base_ward']) & (vehicles['current_ward'].isin(available_wards))].id.values
        else:
            idle_vehicle_wards = vehicles.loc[(vehicles['type'] == 'amb') & (vehicles['status'] == 'idle') & (vehicles['current_ward'] == vehicles['base_ward'])]['base_ward'].values
            # find the wards where there are enough staff and vehicles that are idle
            available_wards = np.intersect1d(staff_wards, idle_vehicle_wards)
            # get IDs of vehicles where status is idle, current ward is base ward and current ward is in available_wards
            idle_at_base_ids = vehicles.loc[(vehicles['type'] == 'amb') & (vehicles['status'] == 'idle') & (vehicles['current_ward'] == vehicles['base_ward']) & (vehicles['current_ward'].isin(available_wards))].id.values
    else:
        idle_at_base_ids = np.array([])
    # for P1 calls, we can re-route. so we need to find vehicles that are en-route and have call priority != p1
    if allow_single and require_ip and require_amb:
        enroute_vehicle_ids = vehicles.loc[(vehicles['type'] == 'amb') & (vehicles['status'] != 'occupied') & (vehicles['call_priority'] != 'p1') & ((vehicles['staff_1'] == 'ip') | (vehicles['staff_2'] == 'ip'))]['id'].values
    elif allow_single and require_ip and not require_amb:
        enroute_vehicle_ids = vehicles.loc[(vehicles['status'] != 'occupied') & (vehicles['call_priority'] != 'p1') & ((vehicles['staff_1'] == 'ip') | (vehicles['staff_2'] == 'ip'))]['id'].values
    elif allow_single and not require_ip and require_amb:
        enroute_vehicle_ids = vehicles.loc[(vehicles['type'] == 'amb') & (vehicles['status'] != 'occupied') & (vehicles['call_priority'] != 'p1') & ((vehicles['staff_1'] != 'NA') | (vehicles['staff_2'] != 'NA'))]['id'].values
    elif allow_single and not require_ip and not require_amb:
        enroute_vehicle_ids = vehicles.loc[(vehicles['status'] != 'occupied') & (vehicles['call_priority'] != 'p1') & ((vehicles['staff_1'] != 'NA') | (vehicles['staff_2'] != 'NA'))]['id'].values
    elif not allow_single and require_ip and require_amb:
        enroute_vehicle_ids = vehicles.loc[(vehicles['type'] == 'amb') & (vehicles['status'] != 'occupied') & (vehicles['call_priority'] != 'p1') & ((vehicles['staff_1'] == 'ip') | (vehicles['staff_2'] == 'ip')) & (vehicles['staff_1'] != 'NA') & (vehicles['staff_2'] != 'NA')]['id'].values
    elif not allow_single and require_ip and not require_amb:
        enroute_vehicle_ids = vehicles.loc[(vehicles['status'] != 'occupied') & (vehicles['call_priority'] != 'p1') & ((vehicles['staff_1'] == 'ip') | (vehicles['staff_2'] == 'ip')) & (vehicles['staff_1'] != 'NA') & (vehicles['staff_2'] != 'NA')]['id'].values
    elif not allow_single and not require_ip and require_amb:
        enroute_vehicle_ids = vehicles.loc[(vehicles['type'] == 'amb') & (vehicles['status'] != 'occupied') & (vehicles['call_priority'] != 'p1') & (vehicles['staff_1'] != 'NA') & (vehicles['staff_2'] != 'NA')]['id'].values
    else:
        enroute_vehicle_ids = vehicles.loc[(vehicles['status'] != 'occupied') & (vehicles['call_priority'] != 'p1') & (vehicles['staff_1'] != 'NA') & (vehicles['staff_2'] != 'NA')]['id'].values

    # now if the demand type is not p1, we need to get the intersection of enroute_vehicle_ids and vehicles that have call priority NA
    if demand_type != ['p1']:
        enroute_vehicle_ids = np.intersect1d(enroute_vehicle_ids, vehicles.loc[vehicles['call_priority'] == 'NA']['id'].values)
    vehicle_ids = np.concatenate((idle_at_base_ids, enroute_vehicle_ids))
    vehicles_filtered = vehicles.loc[vehicles['id'].isin(vehicle_ids)]
    
    # remove duplicates
    vehicles_filtered = vehicles_filtered.drop_duplicates()
    # if ppt in demand types and there's a ptv in the vehicles_filtered df, remove the ambulances
    if 'ppt' in demand_type and 'ptv' in vehicles_filtered['type'].values:
        vehicles_filtered = vehicles_filtered.loc[vehicles_filtered['type'] != 'amb']

    # return vehicles df
    return vehicles_filtered, idle_at_base_ids, allow_single


# function given ward id, returns ward centroid coordinates
def get_ward_centroid(ward_id):
    return wards_centroids.loc[wards_centroids['ward_id'] == ward_id, ['lon', 'lat']].values[0]

# function to set up dispatch_points data frame given dispatch_pts, n_amb, n_ptv, amb_wards, ptv_wards, 
# n_sp, n_ip, sp_wards, ip_wards
def set_up_dispatch_pts(dispatch_pts, n_amb, n_ptv, n_sp, n_ip):
    # note that n_amb, n_ptv, n_sp, n_ip dictionaries structured as follows
    # {ward_id: number_of_vehicles or staff}
    # first set everything to 0
    # convert back to dictionaries
    n_amb = json.loads(n_amb)
    n_ptv = json.loads(n_ptv)
    n_sp = json.loads(n_sp)
    n_ip = json.loads(n_ip)

    dispatch_pts['n_amb'] = 0
    dispatch_pts['n_ptv'] = 0
    # and staff members
    dispatch_pts['n_sp'] = 0
    dispatch_pts['n_ip'] = 0
    # then set the values for the dispatch points
    for i in range(len(dispatch_pts)):
        # get the ward id
        ward_id = str(dispatch_pts.loc[i, 'ward_id'])
        # set the number of ambulances
        if ward_id in n_amb.keys():
            dispatch_pts.loc[i, 'n_amb'] = n_amb[ward_id]
        # set the number of ptvs
        if ward_id in n_ptv.keys():
            dispatch_pts.loc[i, 'n_ptv'] = n_ptv[ward_id]
        # set the number of sp
        if ward_id in n_sp.keys():
            dispatch_pts.loc[i, 'n_sp'] = n_sp[ward_id]
        # set the number of ip
        if ward_id in n_ip.keys():
            dispatch_pts.loc[i, 'n_ip'] = n_ip[ward_id]
    return dispatch_pts


# function that returns the current coordinates of a vehicle, using its 
# approximated route path and percentage of that path completed
def current_coordinates(from_wards, to_wards, times_until_arrival, times_to_destination):
    # convert from_wards and to_wards to pandas
    from_wards = pd.Series(from_wards)
    to_wards = pd.Series(to_wards)
    # empty list to store current coordinates
    current_lats = np.zeros(len(times_until_arrival))
    current_longs = np.zeros(len(times_until_arrival))
    # get the coordinates of the from_ward
    # based on the ward_id, get the coordinates of the ward centroids with correct ward_id using lambda
    # lambda operation
    from_coords = from_wards.apply(lambda x: get_ward_centroid(x))
    to_coords = to_wards.apply(lambda x: get_ward_centroid(x))
    # get the coordinates of the to_ward
    # if time_to_destination is 0, then return to_coords
    # calculate proportion of journey completed
    journey_completed = 1-(times_until_arrival/times_to_destination)
    # calculate the current coordinates. don't use lambda
    # using pandas:
    from_lats = from_coords.apply(lambda ward: ward[1])
    from_longs = from_coords.apply(lambda ward: ward[0])
    to_lats = to_coords.apply(lambda ward: ward[1])
    to_longs = to_coords.apply(lambda ward: ward[0])

    current_lats = from_lats + journey_completed*(to_lats - from_lats)
    current_longs = from_longs + journey_completed*(to_longs - from_longs)

    return current_lats, current_longs

# function that alters vehicle_df to set the status of a vehicle to 'en-route to base' and updates
# the time_until_arrival, time_to_destination and time_until_free columns
def return_to_base(vehicle_id, vehicle_df, travel_time_matrix, travel_time_scaling):
    # get the current location of the vehicle
    current_ward = int(vehicle_df.loc[vehicle_df['id'] == vehicle_id]['current_ward'].values[0])
    # get the base ward location of the vehicle
    base_ward = vehicle_df.loc[vehicle_df['id'] == vehicle_id]['base_ward'].values[0]
    # get the time to travel from current_ward to base_ward
    time_to_base = travel_time(from_ward=current_ward, to_ward=base_ward, travel_time_matrix=travel_time_matrix, scaling=travel_time_scaling)
    # update time_until_arrival, time_to_destination and time_until_free
    # update the vehicle_df, setting these variables all to time_to_base 
    vehicle_df.loc[vehicle_df['id'] == vehicle_id, ['time_until_arrival', 'time_to_destination', 'time_until_free']] = time_to_base
    vehicle_df.loc[vehicle_df['id'] == vehicle_id, 'call_priority'] = 'NA'
    vehicle_df.loc[vehicle_df['id'] == vehicle_id, 'patient_id'] = 'NA'
    # update to_ward
    vehicle_df.loc[vehicle_df['id'] == vehicle_id, 'to_ward'] = base_ward
    # update from_ward
    vehicle_df.loc[vehicle_df['id'] == vehicle_id, 'from_ward'] = current_ward
    # set status to 'en-route'
    vehicle_df.loc[vehicle_df['id'] == vehicle_id, 'status'] = 'en-route to base'
    # return the updated vehicle_df
    return vehicle_df


# function pre-sample calls, and set up queue, staff and vehicle inventory data frames
def pre_sample(n_ticks, dispatch_pts, rates_p1, rates_p2, rates_ppt):
    observed_calls = sample_calls(n_ticks, rates_p1, rates_p2, rates_ppt)
    # initialise queue data frame
    queue = pd.DataFrame({
        'demand_id': np.array([], dtype='int'),
        'patient_id': np.array([], dtype='int'),
        'wait_time': np.array([], dtype='int'),
        'triage_classification': np.array([], dtype='str'),
        'demand_type': [],
        'status': [],
        'ft' : [],
        'closest_central_ward': [],
        'bypassed': [],
        'vehicle_travel_time': np.array([], dtype='float'),
        'old_wait_time': np.array([], dtype='int'),
        'to_ward': np.array([], dtype='int'),
        'vehicle_type': [],
        'staff_1_type': [],
        'staff_2_type': [],
        'tick': np.array([], dtype='int'),
    })

    # initialise vehicle inventory data frame
    bases = np.concatenate((np.repeat(dispatch_pts.ward_id, dispatch_pts.n_amb.astype(int)),
                            np.repeat(dispatch_pts.ward_id, dispatch_pts.n_ptv.astype(int))))

    # count the number of vehicles, to get the row number right
    n_vehicles = sum(dispatch_pts.n_amb) + sum(dispatch_pts.n_ptv)

    # initialise vehicle data frame
    vehicles = pd.DataFrame({
        'id': range(n_vehicles),
        'type': np.concatenate((['amb'] * sum(dispatch_pts.n_amb), ['ptv'] * sum(dispatch_pts.n_ptv))),
        'patient_id': ['NA'] * n_vehicles,
        
        'base_ward': bases,
        
        'from_ward': bases,
        'to_ward': bases,

        'incident_ward': bases,
        'dropoff_ward': bases,
        'current_ward': bases,

        # for the entire trip:
        'time_until_free': np.zeros(n_vehicles),
        # for this segment of the trip:
        'time_until_arrival': np.zeros(n_vehicles),
        # this is fixed:
        'time_to_destination': np.zeros(n_vehicles),
        
        'status': ['idle'] * n_vehicles,
        'call_priority': ['NA'] * n_vehicles,
        # qualifications of the staff members assigned to each vehicle
        'staff_1': ['NA'] * n_vehicles,
        'staff_2': ['NA'] * n_vehicles,
    })

    # use the ils, als and bls columns in dispatch_pts to create a staff register with
    # staff id, base, vehicle_id. initialise vehicle id to 0
    staff_bases = np.concatenate((np.repeat(dispatch_pts.ward_id, dispatch_pts.n_ip.astype(int)),
                                    np.repeat(dispatch_pts.ward_id, dispatch_pts.n_sp.astype(int))))

    # initialise staff data frame
    staff = pd.DataFrame({
        'id': range(sum(dispatch_pts.n_sp) + sum(dispatch_pts.n_ip)),
        'type': np.concatenate((['ip']*sum(dispatch_pts.n_ip), ['sp']*sum(dispatch_pts.n_sp))),
        'status': ['idle']*len(staff_bases),
        'base': staff_bases,
        'vehicle_id': ['NA']*len(staff_bases)
    })

    return observed_calls, queue, vehicles, staff

# function to dispatch the nearest available vehicle
def dispatch_vehicle(available_vehicles_df, idle_at_base_ids, allow_single, 
                    demand_type, demand_queue, vehicle_list, staff_list,
                    mean_wait_times, time_buffers, travel_time_matrix, 
                    amb_travel_scaling, ptv_travel_scaling, ft_communication_prob,
                    restricted_vehicle_movement=False):
    # extract wait times for each demand type from mean_wait_times
    p1_scene_mean = float(mean_wait_times['p1_scene_mean'])
    p2_scene_mean = float(mean_wait_times['p2_scene_mean'])
    ppt_scene_mean = float(mean_wait_times['ppt_scene_mean'])
    facility_mean = float(mean_wait_times['facility_mean'])

    # extract buffers
    scene_buffer = float(time_buffers['scene_buffer'])
    facility_buffer = float(time_buffers['facility_buffer'])

    # decide to which patient a vehicle should be dispatched
    # get row in queue of patient of correct type, with status 'waiting', and maximum wait_time
    # if there are multiple patients with the same wait_time, choose the one with the lowest demand_id
    # reset index of demand_queue
    demand_queue = demand_queue.reset_index(drop=True)
    queue_row_number = demand_queue.loc[(demand_queue['demand_type'].isin(demand_type)) & 
                                        (demand_queue['status'] == 'waiting') &
                                        (demand_queue['bypassed'] != 'yes')].sort_values(by=['wait_time', 'demand_id'], ascending=[False, True]).index[0]

    # get closest central ward
    closest_central_ward = demand_queue.loc[queue_row_number]['closest_central_ward'] 
    if restricted_vehicle_movement:
        # print unique base wards:
        available_vehicles_df = available_vehicles_df.loc[available_vehicles_df['base_ward'] == closest_central_ward]
    if len(available_vehicles_df) == 0:
        # set patient's 'bypassed' to 'yes
        demand_queue.loc[queue_row_number, 'bypassed'] = 'yes'
        return vehicle_list, demand_queue, staff_list
    # if closest_central_ward == '51':
    #     wait_time_scaling = 0.8
    # elif closest_central_ward == '58':
    #     wait_time_scaling = 1
    # elif closest_central_ward == '34':
    #     wait_time_scaling = 1.1
    # else:
    #     wait_time_scaling = 1.1
    # get demand point patient_ward number
    demand_pt_id = demand_queue.loc[queue_row_number]['demand_id']
    # current ward ids of the available ambulances
    vehicle_ids = available_vehicles_df.current_ward.to_list()
     # id of the closest ambulance
    # if len(amb_ids) > 0 and len(ptv_ids) > 0:
    #     amb_prob = 0.5

    closest_ward_id = closest_dispatch(demand_pt_id, vehicle_ids, travel_time_matrix)
    
    # get the index of the first available vehicle with ward_id == closest_ward_id
    closest_vehicle_id = available_vehicles_df[available_vehicles_df.current_ward == closest_ward_id].id.to_list()[0]
 
    if vehicle_list.loc[closest_vehicle_id, 'type'] == 'ptv':
        travel_time_scaling = ptv_travel_scaling
    else:
        travel_time_scaling = amb_travel_scaling
    
    vehicle_trip_time = travel_time(from_ward=closest_ward_id, to_ward=demand_pt_id, travel_time_matrix=travel_time_matrix, scaling=travel_time_scaling)
    
    # add to vehicle_travel_time in queue
    demand_queue.loc[queue_row_number, 'vehicle_travel_time'] = vehicle_trip_time
    # in demand queue update vehicle_type
    demand_queue.loc[queue_row_number, 'vehicle_type'] = vehicle_list.loc[closest_vehicle_id, 'type']
    # set patient's old_wait_time to wait_time
    demand_queue.loc[queue_row_number, 'old_wait_time'] = demand_queue.loc[queue_row_number, 'wait_time']
    # set patient's status to 'dispatched'
    demand_queue.loc[queue_row_number, 'status'] = 'dispatched'
    # calculate the travel time to the closest ward
    # if this vehicle is a ptv
    # if closest vehicle is idle at base:
    if closest_vehicle_id in idle_at_base_ids:
        if len(staff_list[staff_list['vehicle_id'] == closest_vehicle_id]) > 0:
            # extract qualifications of these staff members
            vehicle_list.loc[closest_vehicle_id, 'staff_1'] = staff_list[staff_list['vehicle_id'] == closest_vehicle_id].iloc[0]['type']
            if len(staff_list[staff_list['vehicle_id'] == closest_vehicle_id]) > 1:
                vehicle_list.loc[closest_vehicle_id, 'staff_2'] = staff_list[staff_list['vehicle_id'] == closest_vehicle_id].iloc[1]['type']
        else:
            # extract the staff members that are idle at the base of the closest vehicle
            staff_at_base = staff_list[(staff_list['base'] == vehicle_list.loc[closest_vehicle_id]['base_ward']) & (staff_list['status'] == 'idle')]

            # if we have a P1 call and P1 calls need to have at least 1 IP:
        #     if demand_type == ['p1'] and 'p1_supervised' in staff_rules:
        #         # at least one staff member must be IP, so we set the first staff member to IP
        #         vehicle_list.loc[closest_vehicle_id, 'staff_1'] = 'ip'
        #         # extract staff id of staff_at_base that is 'ip'
        #         staff_1_id = staff_at_base[staff_at_base['type'] == 'ip'].iloc[0]['id']
        #         # set this staff member's status to 'en-route'
        #         staff_list.loc[staff_1_id, 'status'] = 'en-route'
        #         # and update vehicle_id
        #         staff_list.loc[staff_1_id, 'vehicle_id'] = closest_vehicle_id
        
        #         # update staff_at_base to remove this staff member using id
        #         staff_at_base = staff_at_base[staff_at_base['id'] != staff_1_id]

        #         if not allow_single:
        #             # set staff_2 to a random staff member 'idle' at base
        #             staff_2_id = staff_at_base.sample(1).iloc[0]['id']
        #             # in vehicle_list, set staff_2 to the type of this staff member
        #             vehicle_list.loc[closest_vehicle_id, 'staff_2'] = staff_list.loc[staff_2_id, 'type']
        #             # set this staff member's status to 'en-route'
        #             staff_list.loc[staff_2_id, 'status'] = 'en-route'
        #             # and update vehicle_id
        #             staff_list.loc[staff_2_id, 'vehicle_id'] = closest_vehicle_id
        #         else:
        #             # set staff_2 to 'NA'
        #             vehicle_list.loc[closest_vehicle_id, 'staff_2'] = 'NA'
        #         # vehicle_list.loc[closest_vehicle_id, 'call_priority'] = demand_type[0]
        # # if not P1 or we don't need to have at least 1 IP:
        #     else:
                # set staff_1 to a random staff member 'idle' at base
            ip_ids = staff_at_base[staff_at_base['type'] == 'ip'].id.values
            all_ids = staff_at_base.id.values
            if len(ip_ids) > 0 and demand_type == ['p1']:  
                staff_1_id = np.random.choice(ip_ids)
            else:
                staff_1_id = np.random.choice(all_ids)
            # get type of this staff member and update vehicle_list
            vehicle_list.loc[closest_vehicle_id, 'staff_1'] = staff_list.loc[staff_1_id, 'type']
            # set this staff member's status to 'en-route' and update vehicle_id
            staff_list.loc[staff_1_id, 'status'] = 'en-route'
            staff_list.loc[staff_1_id, 'vehicle_id'] = closest_vehicle_id
            if not allow_single:
                # remove this staff member from staff_at_base
                staff_at_base = staff_at_base[staff_at_base['id'] != staff_1_id]
                ip_ids = staff_at_base[staff_at_base['type'] == 'ip'].id.values
                all_ids = staff_at_base.id.values
                # randomly sample another staff member's id
                if len(ip_ids) > 0 and demand_type == ['p1']: 
                    staff_2_id = np.random.choice(ip_ids)
                else:
                    staff_2_id = np.random.choice(all_ids)
                # update staff_2 to match the type of this staff member
                vehicle_list.loc[closest_vehicle_id, 'staff_2'] = staff_list.loc[staff_2_id, 'type']
                # set this staff member's status to 'en-route' and update vehicle_id
                staff_list.loc[staff_2_id, 'status'] = 'en-route'
                # and update vehicle_id
                staff_list.loc[staff_2_id, 'vehicle_id'] = closest_vehicle_id
            else:
                # set staff_2 to 'NA'
                vehicle_list.loc[closest_vehicle_id, 'staff_2'] = 'NA'
                # set vehicle's priority
                # vehicle_list.loc[closest_vehicle_id, 'call_priority'] = demand_type[0]
                #print('staff 1 {} and staff 2 {}'.format(vehicle_list.loc[closest_vehicle_id, 'staff_1'], vehicle_list.loc[closest_vehicle_id, 'staff_2']))
    # if the patient id is numeric, then we need to update the patient's status to 'waiting'
    # and set the patient's vehicle_id to 'NA'

    patient_id = vehicle_list.loc[closest_vehicle_id, 'patient_id']
    # update the patient's status to 'waiting'
    if len(demand_queue.loc[demand_queue.patient_id == patient_id, :]) > 0:
        # set patient's vehicle_travel_time to 0
        demand_queue.loc[demand_queue.patient_id == patient_id, 'vehicle_travel_time'] = 0
        # set the patient's vehicle ID to 'NA'
        demand_queue.loc[demand_queue.patient_id == patient_id, 'vehicle_id'] = 'NA'
        # set the patient's status to 'waiting'
        demand_queue.loc[demand_queue.patient_id == patient_id, 'status'] = 'waiting'
        

    # set this vehicle's incident_ward to demand_pt_id
    vehicle_list.loc[vehicle_list.id == closest_vehicle_id, 'incident_ward'] = demand_pt_id
    # set this vehicle's to_ward to demand_pt_id
    vehicle_list.loc[vehicle_list.id == closest_vehicle_id, 'to_ward'] = demand_pt_id
    # set this vehicle's patient_id to the id of the patient in the queue
    vehicle_list.loc[vehicle_list.id == closest_vehicle_id, 'patient_id'] = demand_queue.iloc[queue_row_number].patient_id

    # set this vehicle's status to 'en-route'
    vehicle_list.loc[vehicle_list.id == closest_vehicle_id, 'status'] = 'en-route'
    # set this vehicle's call_priority to patient's call_priority
    vehicle_list.loc[vehicle_list.id == closest_vehicle_id, 'call_priority'] = demand_type[0]

    # update base for vehicle using demand_pt_id, demand_type and the appropriate 
    # pre-sampled list of destination wards
    # set vehicle's time_to_destination to vehicle_trip_time (we'll increment this again later)
    vehicle_list.loc[vehicle_list.id == closest_vehicle_id, 'time_to_destination'] = vehicle_trip_time
    # set vehicle's time_until_arrival to vehicle_trip_time
    vehicle_list.loc[vehicle_list.id == closest_vehicle_id, 'time_until_arrival'] = vehicle_trip_time
    # scene time
    if demand_type == ['p1']:
        destination = sample_facilities(demand_pt_id, 'p1')[0]
        # 20-40 minutes on scene time
        vehicle_trip_time += np.max([0, np.random.uniform(p1_scene_mean-scene_buffer, 
                                                          p1_scene_mean+scene_buffer, 1)[0]])
    elif demand_type == ['p2']:
        destination = sample_facilities(demand_pt_id, 'p2')[0]
        # on-scene time for 
        vehicle_trip_time += np.max([0, np.random.uniform(p2_scene_mean-scene_buffer, 
                                                          p2_scene_mean+scene_buffer, 1)[0]])
    else:
        destination = sample_facilities(demand_pt_id, 'ppt')[0]
        vehicle_trip_time += np.max([0, np.random.uniform(ppt_scene_mean-scene_buffer, 
                                                          ppt_scene_mean+scene_buffer, 1)[0]])


    # add travel time from demand point to dispatch point
    vehicle_trip_time += travel_time(from_ward=demand_pt_id, to_ward=destination, travel_time_matrix=travel_time_matrix, scaling=travel_time_scaling)
    # between 20 and 40 minutes at facility
    vehicle_trip_time += np.max([0, np.random.uniform(facility_mean-facility_buffer, facility_mean+facility_buffer, 1)[0]])
    
    # set vehicle's time_until_free to vehicle_trip_time
    vehicle_list.loc[vehicle_list.id == closest_vehicle_id, 'time_until_free'] = vehicle_trip_time

    # set vehicle's dropoff_ward to destination
    vehicle_list.loc[vehicle_list.id == closest_vehicle_id, 'dropoff_ward'] = destination
    # set demand_queue to_ward to destination
    demand_queue.loc[queue_row_number, 'to_ward'] = destination
    # update vehicle_id in queue
    demand_queue.loc[queue_row_number, 'vehicle_id'] = closest_vehicle_id
    # update patient_id for vehicle
    vehicle_list.loc[vehicle_list.id == closest_vehicle_id, 'patient_id'] = demand_queue.iloc[queue_row_number].patient_id
    # update call_priority for vehicle
    vehicle_list.loc[vehicle_list.id == closest_vehicle_id, 'call_priority'] = demand_queue.iloc[queue_row_number].demand_type
    
    # in demand_queue update the staff_1_type and staff_2_type to the staff types for the closest vehicle
    demand_queue.loc[queue_row_number, 'staff_1_type'] = vehicle_list.loc[vehicle_list.id == closest_vehicle_id, 'staff_1'].values[0]
    demand_queue.loc[queue_row_number, 'staff_2_type'] = vehicle_list.loc[vehicle_list.id == closest_vehicle_id, 'staff_2'].values[0]
    # if the call is ft, with a probability 0.3, we set it to removed, and set vehicle status to 'call ft'
    if demand_queue.loc[queue_row_number, 'ft'] == 'ft':
        if np.random.uniform() < ft_communication_prob:
            demand_queue.loc[queue_row_number, 'status'] = 'removed'
            vehicle_list.loc[vehicle_list.id == closest_vehicle_id, 'status'] = 'call ft'

    return vehicle_list, demand_queue, staff_list

# function to run the sim
# run the sim

# time_to_destination: the fixed time it takes to travel from from_ward to to_ward
# time_until_arrival: the time it takes to travel from current position to to_ward
# time_until_free: the time it takes to travel from current position to dropoff_ward
# ft_rates: [ft_p1, ft_p2, ft_ppt]

def run_sim(iters, 
            dispatch_pts, 
            travel_time_matrix=gmaps_travel_time_matrix,
            rates_p1=rates_p1, 
            rates_p2=rates_p2, rates_ppt=rates_ppt,
            mean_wait_times='', time_buffers='',
            ptv_travel_scaling=1, amb_travel_scaling=1,
            modelled_ft=True, verbose=False, outputs=[], 
            restricted_vehicle_movement=False,
            prob_amb='', prob_ip='', prob_single_staff='', pb=True,
            prioritise_p1_prob=0, prioritise_p2_prob=0, 
            ft_communication_prob=0, priority_correction_prob=0, 
            reroute_prob=1):

    # if mean wait times and time buffers are not provided, use return 'missing inputs'
    if mean_wait_times == '' or time_buffers == '' or prob_single_staff == '':
        return 'missing inputs'
    # apply json loads to read in dictionaries
    mean_wait_times = json.loads(mean_wait_times)
    time_buffers = json.loads(time_buffers)
    prob_single_staff = json.loads(prob_single_staff)
    prob_amb = json.loads(prob_amb)
    prob_ip = json.loads(prob_ip)

    # pre-sample calls and set up vehicles, staff, queue using pre_sample(n_iters, dispatch_pts)
    observed_calls, queue, vehicles, staff = pre_sample(iters, dispatch_pts, rates_p1, rates_p2, rates_ppt)
    # initialise outputs data frame
    if 'patient' in outputs:
        patient_outputs = pd.DataFrame(columns=['patient_id', 'patient_type', 
                                                'patient_wait_time', 'queue_time', 'dispatch_time',
                                                'triage_classification',
                                                'patient_ward', 'tick_call_placed', 'demand_type', 'ft', 
                                                'vehicle_id'])
    if 'system' in outputs:
        system_outputs = pd.DataFrame(columns=['queue_length', 'queue_length_p1', 'queue_length_p2', 
                                            'queue_length_ppt',
                                            'amb_idle', 'amb_enroute', 'amb_occupied', 
                                            'ptv_idle', 'ptv_enroute', 'ptv_occupied',
                                            'ip_idle', 'ip_enroute',
                                            'sp_idle', 'sp_enroute',
                                            'ptv_single', 'amb_single'])
        
    if 'vehicle_locations' in outputs:
        vehicle_locations = pd.DataFrame(columns=['tick', 'vehicle_id', 'vehicle_type', 'vehicle_lat', 'vehicle_lon', 'vehicle_base'])
    
    types = ['p1', 'p2', 'ppt']
    max_patient_id = 0
    ids_to_remove = []
    ft_ids = []
    p1_single = 0
    nonp1_single = 0
    # iterate through ticks
    # progress bar 
    if pb == True:
        pbar = tqdm(total=iters)
    for tick in range(iters):
        # reroute_p1
        reroute_p1 = False
        if np.random.uniform() < reroute_prob:
            reroute_p1 = True

        # set all bypassed variable in queue to ''
        queue.loc[:, 'bypassed'] = ''
        # update progress bar
        if pb == True:
            pbar.update(1)
        # update system outputs
        # 'queue_length', 'queue_length_p1', 'queue_length_p2', 'queue_length_ppt',
        # 'amb_idle', 'amb_enroute', 'amb_occupied', 
        # 'ptv_idle', 'ptv_enroute', 'ptv_occupied'
        # queue length
        if 'system' in outputs:
            system_outputs.loc[tick, 'queue_length'] = len(queue)
            system_outputs.loc[tick, 'queue_length_p1'] = len(queue[queue.demand_type == 'p1'])
            system_outputs.loc[tick, 'queue_length_p2'] = len(queue[queue.demand_type == 'p2'])
            system_outputs.loc[tick, 'queue_length_ppt'] = len(queue[queue.demand_type == 'ppt'])
            # ambulance status
            system_outputs.loc[tick, 'amb_idle'] = len(vehicles[(vehicles.type == 'amb') & (vehicles.status == 'idle')])
            # count number of ambs en-route OR en-route to base
            system_outputs.loc[tick, 'amb_enroute'] = len(vehicles[(vehicles.type == 'amb') & ((vehicles.status == 'en-route') | (vehicles.status == 'en-route to base'))])
            system_outputs.loc[tick, 'amb_occupied'] = len(vehicles[(vehicles.type == 'amb') & (vehicles.status == 'occupied')])
            # ptv status
            system_outputs.loc[tick, 'ptv_idle'] = len(vehicles[(vehicles.type == 'ptv') & (vehicles.status == 'idle')])
            system_outputs.loc[tick, 'ptv_enroute'] = len(vehicles[(vehicles.type == 'ptv') & ((vehicles.status == 'en-route') | (vehicles.status == 'en-route to base'))])
            system_outputs.loc[tick, 'ptv_occupied'] = len(vehicles[(vehicles.type == 'ptv') & (vehicles.status == 'occupied')])
            # ip status, using staff list
            system_outputs.loc[tick, 'ip_idle'] = len(staff[(staff.type == 'ip') & (staff.status == 'idle')])
            system_outputs.loc[tick, 'ip_enroute'] = len(staff[(staff.type == 'ip') & (staff.status == 'en-route')])
            # sp status, using staff list
            system_outputs.loc[tick, 'sp_idle'] = len(staff[(staff.type == 'sp') & (staff.status == 'idle')])
            system_outputs.loc[tick, 'sp_enroute'] = len(staff[(staff.type == 'sp') & (staff.status == 'en-route')])


        # set vehicles with status 'call ft' to 'idle'
        vehicles.loc[vehicles.status == 'call ft', 'status'] = 'idle'
        # reduce time_until_free by 1 for all vehicles
        vehicles.loc[:, 'time_until_free'] = vehicles.loc[:, 'time_until_free'] - 1
        # reduce time_until_arrival by 1 for all vehicles
        vehicles.loc[:, 'time_until_arrival'] = vehicles.loc[:, 'time_until_arrival'] - 1

        # if vehicles have negative time_until_arrival or time_until_free, set to 0
        vehicles.loc[vehicles.time_until_free < 0, 'time_until_free'] = 0
        vehicles.loc[vehicles.time_until_arrival < 0, 'time_until_arrival'] = 0
        
        # extract indices of vehicles that have arrived at their destination and are en-route
        enroute_arrived = (vehicles.time_until_arrival == 0) & (vehicles.patient_id != 'NA')
        ids_to_remove = vehicles.loc[enroute_arrived, 'patient_id']


        # if a vehicle's time_until_arrival is 0 and status is 'en-route':
        # 1. set from_ward to incident_ward
        vehicles.loc[enroute_arrived, 'from_ward'] = vehicles.loc[enroute_arrived, 'incident_ward']
        # 2. set to_ward to dropoff_ward
        vehicles.loc[enroute_arrived, 'to_ward'] = vehicles.loc[enroute_arrived, 'dropoff_ward']
        # 3. set status to 'occupied', because it has picked up its patient
        vehicles.loc[enroute_arrived, 'status'] = 'occupied'
        # if patient's ft variable == 'ft', set status to 'en-route to base'
        for id in vehicles[enroute_arrived].id:
            # get patient_id
            patient_id = vehicles.loc[vehicles.id == id, 'patient_id'].values[0]
            # get patient_type
            patient_type = vehicles.loc[vehicles.id == id, 'call_priority'].values[0]
            # decide if patient is ft or not, if modelled_ft == True
            if modelled_ft == True:
                # filter ft_rates_output so that rt_ticks == patient_wait_time
                ft_rates_filtered = ft_rates_output[ft_rates_output.rt_ticks == np.min([queue[queue.patient_id == patient_id].wait_time.values[0], 100])]
                # and patient_type == patient_type
                ft_rates_filtered = ft_rates_filtered[ft_rates_filtered.patient_type == patient_type]

                # get ft_probability from ft_rates_output
                ft_probability = ft_rates_filtered.fitted.values[0]
                # print('ft_probability: ', ft_probability)
                # get random number between 0 and 1
                if np.random.uniform(0, 1) < ft_probability:
                    queue.loc[queue['patient_id'] == patient_id, 'ft'] = 'ft'
                    # append patient_id to ft_ids
                    ft_ids.append(patient_id)


            # if their patient's ft variable == 'ft', set status to 'en-route to base'
            if queue.loc[queue.patient_id == patient_id, 'ft'].values[0] == 'ft':
                # if vehicle is a ptv, set travel_time_scaling to ptv_travel_time_scaling
                if vehicles.loc[vehicles.id == id, 'type'].values[0] == 'ptv':
                    travel_time_scaling = ptv_travel_scaling
                else:
                    travel_time_scaling = amb_travel_scaling
                vehicles = return_to_base(id, vehicles, travel_time_matrix, travel_time_scaling)


        # 4. set time_to_destination to time_until_free. time_to_destination is the fixed time it 
        # takes to travel from one ward to another
        vehicles.loc[enroute_arrived, 'time_to_destination'] = vehicles.loc[enroute_arrived, 'time_until_free']
        # 5. set time_until_arrival to time_until_free. time_until_arrival is the time remaining, and varies
        vehicles.loc[enroute_arrived, 'time_until_arrival'] = vehicles.loc[enroute_arrived, 'time_until_free']
        # 6: remove patients from queue who have been picked up by vehicles
        # if len(ids_to_remove) > 0:
        #     print('ids_to_remove: {}'.format(ids_to_remove.tolist()))
        # in queue assign status 'removed' to patients who have been picked up by vehicles, using ids_to_remove
        queue.loc[queue.patient_id.isin(ids_to_remove), 'status'] = 'removed'
        # for the ids_to_remove, set wait_time to old wait_time + vehicle_travel_time
        queue.loc[queue.patient_id.isin(ids_to_remove), 'wait_time'] = queue.loc[queue.patient_id.isin(ids_to_remove), 'wait_time'] + queue.loc[queue.patient_id.isin(ids_to_remove), 'vehicle_travel_time'] - queue.loc[queue.patient_id.isin(ids_to_remove), 'vehicle_travel_time'].astype(int)
                # add 1 to patient wait time in queue for patients not removed
        queue.loc[queue.status != 'removed', 'wait_time'] = queue.loc[queue.status != 'removed', 'wait_time'] + 1
        vehicles.loc[enroute_arrived, 'call_priority'] = 'NA'
        # set vehicle_id in patient outputs to the correct vehicle_ids
        patient_outputs.loc[patient_outputs.patient_id.isin(ids_to_remove), 'vehicle_id'] = vehicles.loc[enroute_arrived, 'id'].values
        vehicles.loc[enroute_arrived, 'patient_id'] = 'NA'
       
        # extract the indices of vehicles that are en-route to base, and have arrived at base
        enroute_base_arrived = (vehicles.current_ward == vehicles.base_ward) & (vehicles.status == 'en-route to base') & (vehicles.to_ward == vehicles.base_ward)
        # for these vehicles, 
        # 1. set time_until_arrival to 0
        vehicles.loc[enroute_base_arrived, 'time_until_free'] = 0
        # 2. set status to 'idle'
        vehicles.loc[enroute_base_arrived, 'status'] = 'idle'
        # 3. set call_priority to 'NA'
        # vehicles.loc[enroute_base_arrived, 'call_priority'] = 'NA'
        # 4. set staff_1 and staff_2 to 'NA'
        vehicles.loc[enroute_base_arrived, 'staff_1'] = 'NA'
        vehicles.loc[enroute_base_arrived, 'staff_2'] = 'NA'
        # 5. set status of staff members who are in these vehicles to 'idle'
        staff.loc[staff.vehicle_id.isin(vehicles.loc[enroute_base_arrived, 'id']), 'status'] = 'idle'
        # set vehicle_id of staff members who are in these vehicles to 'NA'
        staff.loc[staff.vehicle_id.isin(vehicles.loc[enroute_base_arrived, 'id']), 'vehicle_id'] = 'NA'

        # if a vehicle's time_until_free is 0 and it is not at base, it must be returned to base
        # set from_ward to dropoff_ward and to_ward to base_ward
        arrived_occupied = (vehicles.time_until_free == 0) & (vehicles.current_ward != vehicles.base_ward) & (vehicles.status == 'occupied')
        # send back to base using return_to_base function
        # number of vehicles with demand_type == 'p1' and staff_1 == 'NA'
        p1_single = np.append(p1_single, vehicles.loc[((vehicles.call_priority == 'p1') & 
                                                      ((vehicles.staff_1 == 'NA') & (vehicles.staff_2 != 'NA') | (vehicles.staff_1 != 'NA') & (vehicles.staff_2 == 'NA')) &
                                                      vehicles.patient_id != 'NA'), 'patient_id'].tolist())
        nonp1_single = np.append(nonp1_single, vehicles.loc[((vehicles.call_priority != 'p1') &
                                                            ((vehicles.staff_1 == 'NA') & (vehicles.staff_2 != 'NA') | (vehicles.staff_1 != 'NA') & (vehicles.staff_2 == 'NA')) &
                                                            vehicles.patient_id != 'NA'), 'patient_id'].tolist()) 
        # make unique
        p1_single = np.unique(p1_single)
        nonp1_single = np.unique(nonp1_single)
        # append the lengths of these to system_outputs
        system_outputs.loc[tick, 'p1_single'] = len(p1_single)
        system_outputs.loc[tick, 'nonp1_single'] = len(nonp1_single)
        # iterate through these vehicles and send them home
        for vehicle_id in vehicles.loc[arrived_occupied, 'id']:
            if vehicles.loc[vehicles.id == vehicle_id, 'type'].values[0] == 'ptv':
                travel_time_scaling = ptv_travel_scaling
            else:
                travel_time_scaling = amb_travel_scaling
            vehicles = return_to_base(vehicle_id, vehicles, travel_time_matrix, travel_time_scaling)
        
        # set vehicle status to 'idle' if time_until_free = 0
        vehicles.loc[vehicles.time_until_free == 0, 'status'] = 'idle'
        # set call_priority to 'NA'
        vehicles.loc[vehicles.time_until_free == 0, 'call_priority'] = 'NA'
        # set patient_id to 'NA'
        vehicles.loc[vehicles.time_until_free == 0, 'patient_id'] = 'NA'

        # extract vehicles in motion (time_until_arrival > 0)
        # we need to update their coordinates 
        # get their indices
        motion_indices = vehicles.status != 'idle'
        motion_vehicles = vehicles.loc[motion_indices, :]
  
        # time this loop
        if len(motion_vehicles) > 0:
            # calculate the current latitude and longitude
            # of the vehicle
            current_lats, current_longs = current_coordinates(motion_vehicles['from_ward'], 
                                                            motion_vehicles['to_ward'], 
                                                            motion_vehicles['time_until_arrival'], 
                                                            motion_vehicles['time_to_destination'])
            if 'vehicle_locations' in outputs:
                df_to_append = pd.DataFrame({'tick': [tick]*len(motion_vehicles), 'vehicle_id': motion_vehicles.id, 
                                             'current_lat': current_lats, 'current_lon': current_longs, 'vehicle_base': motion_vehicles.base_ward})
                vehicle_locations = pd.concat([vehicle_locations, df_to_append], axis=0)
           
            vehicles.loc[motion_indices, 'current_ward'] = get_ward(wards_shp, lats=current_lats, longs=current_longs)


        types_nonzero = np.nonzero(np.sum(observed_calls[:, :, tick], axis=1))[0]
        locations_nonzero = np.nonzero(np.sum(observed_calls[:, :, tick], axis=0))[0]
        # add calls to queue!:
        if len(types_nonzero) > 0:
            for type_index in types_nonzero:
                for location_index in locations_nonzero:
                    # number of new calls of type i at location j
                    n_new_calls = observed_calls[type_index, location_index, tick]
                    if n_new_calls > 0:
                        # set call type
                        calltype = types[type_index]
                        if calltype == 'p1':
                            # find demand ward id
                            ward = p1_dist.from_ward[location_index]
                        elif calltype == 'p2':
                            ward = p2_dist.from_ward[location_index]
                        else:
                            ward = ppt_dist.from_ward[location_index]
                        # add to queue
                        new_row = pd.DataFrame({
                                'demand_id': np.array([ward] * n_new_calls, dtype='int'),
                                'wait_time': [0] * n_new_calls,
                                'status': ['waiting'] * n_new_calls,
                                'demand_type': [calltype] * n_new_calls,
                                'triage_classification': ['NA'] * n_new_calls,
                                # set patient IDs to the max patient ID in patient_outputs 
                                'patient_id': range(max_patient_id + 1, 
                                                    max_patient_id + n_new_calls + 1),
                                'ft': [''] * n_new_calls,
                                'closest_central_ward': ['NA'] * n_new_calls,
                                'bypassed': [''] * n_new_calls,
                                'tick': [tick] * n_new_calls
                            })

                        # with probability scramble_prob, set demand_type to 'p2' for p1 calls
                        for index, row in new_row.iterrows():
                            # determine triage classification based on demand type:
                            random1 = np.random.uniform()
                            random2 = np.random.uniform()
                            demand_type = new_row.loc[index, 'demand_type']
                            if demand_type == 'p1':
                                if random1 < 0.726:
                                    new_row.loc[index, 'triage_classification'] = 'yellow'
                                elif random1 >= 0.726 and random1 < (0.726 + 0.058):
                                    new_row.loc[index, 'triage_classification'] = 'green'
                                else:
                                    new_row.loc[index, 'triage_classification'] = 'red'

                            elif demand_type == 'p2':
                                if random1 < 0.695:
                                    new_row.loc[index, 'triage_classification'] = 'yellow'
                                elif random1 >= 0.695 and random1 < (0.695 + 0.276):
                                    new_row.loc[index, 'triage_classification'] = 'green'
                                else:
                                    new_row.loc[index, 'triage_classification'] = 'red'
                            else:
                                if random1 < 0.055:
                                    new_row.loc[index, 'triage_classification'] = 'yellow'
                                elif random1 >= 0.055 and random1 < (0.055 + 0.942):
                                    new_row.loc[index, 'triage_classification'] = 'green'
                                else:
                                    new_row.loc[index, 'triage_classification'] = 'red'
                            # now we re-prioritise
                            if random2 < priority_correction_prob:
                                if new_row.loc[index, 'triage_classification'] == 'red' and demand_type != 'ppt':
                                    new_row.loc[index, 'demand_type'] = 'p1'
                                elif new_row.loc[index, 'triage_classification'] == 'yellow' and demand_type != 'ppt':
                                    new_row.loc[index, 'demand_type'] = 'p2'
                                elif new_row.loc[index, 'triage_classification'] == 'green' and demand_type != 'ppt':
                                    new_row.loc[index, 'demand_type'] = 'p2'
                        
                            # set closest central ward
                            new_row.loc[index, 'closest_central_ward'] = get_area(row['demand_id'])
                        # append 'patient_id', 'patient_type', 'patient_wait_time', 
                        # 'patient_ward', 'tick_call_placed' in patient_outputs
                        # make row to append
                        if 'patient' in outputs:
                            new_patient_row = pd.DataFrame({
                                    'patient_id': range(max_patient_id + 1, 
                                                        max_patient_id + n_new_calls + 1),
                                    'patient_type': new_row['demand_type'],
                                    'patient_wait_time': [0] * n_new_calls,
                                    'queue_time': [0] * n_new_calls,
                                    'dispatch_time': [0] * n_new_calls,
                                    'patient_ward': new_row['demand_id'],
                                    'triage_classification': new_row['triage_classification'],
                                    'tick_call_placed': [tick] * n_new_calls,
                                    'ft': new_row['ft']
                                })
                            # append to patient_outputs using pd.concat
                            patient_outputs = pd.concat([patient_outputs, new_patient_row])
                        queue = pd.concat([queue, new_row]).reset_index(drop=True)

                        max_patient_id = int(max_patient_id + n_new_calls)
            
        # update current_ward using to_ward, from_ward
        queue_start = len(queue)
        # set vehicle status to idle if time_until_free = 0
        vehicles.loc[vehicles.time_until_free == 0, 'status'] = 'idle'
        # # set patient_id to NA
        # vehicles.loc[vehicles.time_until_free == 0, 'patient_id'] = 'NA'
        # # set call priority to NA
        # vehicles.loc[vehicles.time_until_free == 0, 'call_priority'] = 'NA'

        # only do something if the queue isn't empty and there are not 0 observed calls
        # and there are vehicles available (status is not occupied)

        avail_veh, idle_at_base_ids, allow_single = available_vehicles(['ppt'], vehicles, staff, prob_single_staff,
                                                                         prob_amb, prob_ip, reroute_p1)
        
        if verbose == True:
            # print('STAFF')
            # print(staff)
            print('QUEUE')
            # print only patients not removed
            # show columns patient_id  wait_time triage_classification demand_type  closest_central_ward bypassed vehicle_travel_time status
            print(queue.loc[queue.status != 'removed', ['patient_id', 'wait_time', 'triage_classification', 'demand_type', 'closest_central_ward', 'bypassed', 'old_wait_time', 'vehicle_travel_time', 'status']])
            # print('ALL VEHICLES:')
            # print(vehicles)
            # print('AVAILABLE VEHICLES:')
            # print(avail_veh)
        
        if queue_start != 0 and len(avail_veh) != 0:
            # show patient_outputs where id is in queue
          # count number of available ambulances
            # count number of p1 patients in queue (demand_type == 'p1') and status == 'waiting
            n_p1_queue = len(queue.loc[(queue.demand_type == 'p1') & (queue.status == 'waiting'), :])
            # count number of p2 patients in queue (demand_type == 'p2') and status == 'waiting
            n_p2_queue = len(queue.loc[(queue.demand_type == 'p2') & (queue.status == 'waiting'), :])
            # count number of ppt patients in queue (demand_type == 'ppt') and status == 'waiting
            n_ppt_queue = len(queue.loc[(queue.demand_type == 'ppt') & (queue.status == 'waiting'), :])

            # if there are p1s in the queue and available ambulances:
            vehicles.loc[vehicles.time_until_free == 0, 'status'] = 'idle'
            

            prioritise_p1 = False
            prioritise_p2 = False
            if np.random.uniform() < prioritise_p1_prob:
                prioritise_p1 = True
            if np.random.uniform() < prioritise_p2_prob:
                prioritise_p2 = True
            
            if prioritise_p1:
                # check whether there are idle staff-vehicle pairs to respond
                avail_veh, idle_at_base_ids, allow_single = available_vehicles(['p1'], vehicles, staff, prob_single_staff,
                                                                            prob_amb, prob_ip, reroute_p1)

                # if there is >=1 ambulance en-route to a non-p1 call,
                # and not both staff_1 and staff_2 are bls, set p1_capacity to True
                while n_p1_queue > 0 and len(avail_veh) > 0:
                    # if there are p1s in the queue and available ambulances:
                    vehicles, queue, staff = dispatch_vehicle(avail_veh, idle_at_base_ids, allow_single, 
                                                            ['p1'], queue,
                                                            vehicles, staff, mean_wait_times, 
                                                            time_buffers, travel_time_matrix, amb_travel_scaling, 
                                                            ptv_travel_scaling, ft_communication_prob,
                                                            restricted_vehicle_movement=restricted_vehicle_movement)
                    n_p1_queue = len(queue.loc[(queue.demand_type == 'p1') & (queue.status == 'waiting') & (queue.bypassed != 'yes'), :])
                    # n_avail_amb = sum((vehicles.type == 'amb') & (vehicles.time_until_free == 0))
                    avail_veh, idle_at_base_ids, allow_single = available_vehicles(['p1'], vehicles, staff, prob_single_staff,
                                                                            prob_amb, prob_ip, reroute_p1)
            
            if prioritise_p2:
                # then empty p2 primary queue
                # n_avail_amb = np.sum((vehicles['type'] == 'amb') & (vehicles['time_until_free'] == 0))
                # if there are p2 primaries in the queue and available ambulances:
                avail_veh, idle_at_base_ids, allow_single = available_vehicles(['p2'], vehicles, staff, prob_single_staff,
                                                                            prob_amb, prob_ip, reroute_p1)
                # count number of staff with status == 'idle'
                while n_p2_queue > 0 and len(avail_veh) > 0:
                    vehicles, queue, staff = dispatch_vehicle(avail_veh, idle_at_base_ids, allow_single, 
                                                            ['p2'], queue,
                                                            vehicles, staff, mean_wait_times, 
                                                            time_buffers, travel_time_matrix, amb_travel_scaling, 
                                                            ptv_travel_scaling, ft_communication_prob,
                                                            restricted_vehicle_movement=restricted_vehicle_movement)
                    n_p2_queue = len(queue.loc[(queue.demand_type == 'p2') & (queue.status == 'waiting') & (queue.bypassed != 'yes'), :])
                    # n_avail_amb = sum((vehicles.type == 'amb') & (vehicles.time_until_free == 0))
                    # # count number of staff with status == 'idle'
                    # n_avail_staff = np.sum((staff['status'] == 'idle'))
                    avail_veh, idle_at_base_ids, allow_single = available_vehicles(['p2'], vehicles, staff, prob_single_staff,
                                                                            prob_amb, prob_ip, reroute_p1)

            # n available of all vehicles
            # n_avail_veh = np.sum((vehicles['time_until_free'] == 0))
            # then empty p2 ppt queue
            # if there are ppts in the queue and available vehicles:
            avail_veh_p1, idle_at_base_ids, allow_single = available_vehicles(['p1'], vehicles, staff, prob_single_staff,
                                                                         prob_amb, prob_ip, reroute_p1)
            avail_veh_p2, idle_at_base_ids, allow_single = available_vehicles(['p2'], vehicles, staff, prob_single_staff,
                                                            prob_amb, prob_ip, reroute_p1)
            avail_veh_ppt, idle_at_base_ids, allow_single = available_vehicles(['ppt'], vehicles, staff, prob_single_staff,
                                                prob_amb, prob_ip, reroute_p1)
            avail_veh = pd.concat([avail_veh_p1, avail_veh_p2, avail_veh_ppt]).drop_duplicates()

            possible_call_types = []
            if len(avail_veh_p1) > 0 and n_p1_queue > 0:
                possible_call_types.append('p1')
            if len(avail_veh_p2) > 0 and n_p2_queue > 0:
                possible_call_types.append('p2')
            if len(avail_veh_ppt) > 0 and n_ppt_queue > 0:
                possible_call_types.append('ppt')
            
            while len(possible_call_types) > 0:
                # get call type of first patient in queue, with maximum wait_time
                row_number = queue.loc[
                    (queue['demand_type'].isin(possible_call_types)) &
                    (queue['status'] == 'waiting') &
                    (queue['bypassed'] != 'yes'),
                ].sort_values(by=['wait_time', 'patient_id'], ascending=[False, True]).index[0]
                
                # Get call_type of the corresponding patient
                call_type = [queue.loc[row_number, 'demand_type']]
                avail_veh, idle_at_base_ids, allow_single = available_vehicles(call_type, vehicles, staff, prob_single_staff,
                                                                        prob_amb, prob_ip, reroute_p1)
                if len(avail_veh) > 0:
                    vehicles, queue, staff = dispatch_vehicle(avail_veh, idle_at_base_ids, allow_single, 
                                                            call_type, queue,
                                                            vehicles, staff, mean_wait_times, 
                                                            time_buffers, travel_time_matrix, amb_travel_scaling, 
                                                            ptv_travel_scaling, ft_communication_prob,
                                                            restricted_vehicle_movement=restricted_vehicle_movement)
                    
                n_p1_queue = len(queue.loc[(queue.demand_type == 'p1') & (queue.status == 'waiting') & (queue.bypassed != 'yes'), :])
                n_p2_queue = len(queue.loc[(queue.demand_type == 'p2') & (queue.status == 'waiting') & (queue.bypassed != 'yes'), :])
                n_ppt_queue = len(queue.loc[(queue.demand_type == 'ppt') & (queue.status == 'waiting') & (queue.bypassed != 'yes'), :])

                avail_veh_p1, idle_at_base_ids, allow_single = available_vehicles(['p1'], vehicles, staff, prob_single_staff,
                                                                                        prob_amb, prob_ip, reroute_p1)
                avail_veh_p2, idle_at_base_ids, allow_single = available_vehicles(['p2'], vehicles, staff, prob_single_staff,
                                                                prob_amb, prob_ip, reroute_p1)
                avail_veh_ppt, idle_at_base_ids, allow_single = available_vehicles(['ppt'], vehicles, staff, prob_single_staff,
                                                    prob_amb, prob_ip, reroute_p1)
                avail_veh = pd.concat([avail_veh_p1, avail_veh_p2, avail_veh_ppt]).drop_duplicates()
                
                possible_call_types = []
                if len(avail_veh_p1) > 0 and n_p1_queue > 0:
                    possible_call_types.append('p1')
                if len(avail_veh_p2) > 0 and n_p2_queue > 0:
                    possible_call_types.append('p2')
                if len(avail_veh_ppt) > 0 and n_ppt_queue > 0:
                    possible_call_types.append('ppt')
            # set all bypassed to ''
            queue.loc[queue.bypassed == 'yes', 'bypassed'] = ''
    # print 'status', 'bypassed', 'patient_id', 'demand_type', 'wait_time' from queue
    if 'patient' in outputs:
        patient_outputs = queue
        # rename column 'wait_time' to 'patient_wait_time'
        patient_outputs.rename(columns={'wait_time': 'patient_wait_time'}, inplace=True)
        # rename demand_type to patient_type
        patient_outputs.rename(columns={'demand_type': 'patient_type'}, inplace=True)
            # patient_outputs = pd.merge(patient_outputs, queue_copy, how='left', on='patient_id')            
            # # waiting_ids: get the ids of patients that are waiting in queue
            # waiting_ids = queue[queue.status == 'waiting'].patient_id.tolist()
            # # dispatched_ids: get the ids of patients that are dispatched in queue
            # dispatched_ids = queue[queue.status == 'dispatched'].patient_id.tolist()
            # # for patients that are waiting, update their queue time by + 1 in patient_outputs
            # patient_outputs.loc[patient_outputs.patient_id.isin(waiting_ids), 'queue_time'] = (
            #                                     patient_outputs.loc[
            #                                     patient_outputs.patient_id.isin(waiting_ids), 'queue_time'] + 1)
            # # for patients that are dispatched, update their dispatch time by + 1 in patient_outputs
            # patient_outputs.loc[patient_outputs.patient_id.isin(dispatched_ids), 'dispatch_time'] = (
            #                                     patient_outputs.loc[
            #                                     patient_outputs.patient_id.isin(dispatched_ids), 'dispatch_time'] + 1)
    
        # set vehicle call_priority to priority from queue using patient_id
        # get ids of vehicles that are not idle

    if modelled_ft == True:
        patient_outputs.loc[patient_outputs.patient_id.isin(ft_ids), 'ft'] = 'ft'
    # for all patient wait times in patient_outputs add u(0, 1) noise
    if 'patient' in outputs:
        patient_outputs['patient_wait_time'] = patient_outputs['patient_wait_time'] 
    if 'patient' in outputs and 'system' in outputs and 'vehicle_locations' in outputs:
        return patient_outputs, system_outputs, vehicle_locations
    elif 'patient' in outputs and 'system' in outputs:
        return patient_outputs, system_outputs
    elif 'system' in outputs and 'patient' not in outputs:
        return system_outputs
    else:
        return patient_outputs       

print(travel_time(from_ward=2, to_ward=31, travel_time_matrix=gmaps_travel_time_matrix, scaling=10*60))
