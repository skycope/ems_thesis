import pandas as pd
import numpy as np
import string
import random

# import model
from model_functions import run_sim, set_up_dispatch_pts
# import data
from data_loading import dispatch_pts_outline, rates_p1, rates_p2, rates_ppt

def weekend_weekday(tick):
    # integer div by 2
    tick = tick // 2
    if tick <= 720: 
        return 'weekday'
    else:  
        return 'weekend'

def main():
    run_id = ''.join(random.choices(string.ascii_uppercase + string.digits, k=10))
    print(f'running scenarios with run id {run_id}...')
    # read in scenarios_df
    sheet_url = "https://docs.google.com/spreadsheets/d/1Urk8KPvZEouzupBZm7YBi60_bBvgJy3O5jjzCZfnMZE/export?format=csv"
    scenarios_df = pd.read_csv(sheet_url, )
    scenario_outputs = pd.DataFrame({
    })
    # this is a for loop iterating over the df
    # rows 31 to 42 are the rows
    for row_number in range(1):
        row_number = 0
        print('running scenario {}'.format(scenarios_df.loc[row_number, 'scenario name']))
        # extract parameter values form scenarios_df
        n_amb = scenarios_df.loc[row_number, 'n_amb'].replace("'", '"')
        n_ptv = scenarios_df.loc[row_number, 'n_ptv'].replace("'", '"')
        n_sp = scenarios_df.loc[row_number, 'n_sp'].replace("'", '"')
        n_ip = scenarios_df.loc[row_number, 'n_ip'].replace("'", '"')
        ptv_travel_scaling = scenarios_df.loc[row_number, 'ptv_travel_scaling']
        amb_travel_scaling = scenarios_df.loc[row_number, 'amb_travel_scaling']
        prob_amb = scenarios_df.loc[row_number, 'prob_amb'].replace("'", '"')
        prob_ip = scenarios_df.loc[row_number, 'prob_ip'].replace("'", '"')
        prob_sp = scenarios_df.loc[row_number, 'prob_sp'].replace("'", '"')
        time_buffers = scenarios_df.loc[row_number, 'time_buffers'].replace("'", '"')
        mean_wait_times = scenarios_df.loc[row_number, 'mean_wait_times'].replace("'", '"')
        prob_single_staff = scenarios_df.loc[row_number, 'prob_single_staff'].replace("'", '"')
        ft_communication_prob = scenarios_df.loc[row_number, 'ft_communication_prob']
        prioritise_p1_prob = scenarios_df.loc[row_number, 'prioritise_p1_prob']
        prioritise_p2_prob = scenarios_df.loc[row_number, 'prioritise_p2_prob']
        reroute_prob = scenarios_df.loc[row_number, 'reroute_prob']
        priority_correction_prob = scenarios_df.loc[row_number, 'priority_correction_prob']
        restricted_vehicle_movement = scenarios_df.loc[row_number, 'restricted_vehicle_movement']
        # convert to bool (it's FALSE or TRUE)
        restricted_vehicle_movement = restricted_vehicle_movement == 'TRUE'
        # print out parameters
        print(f'running scenario: {scenarios_df.loc[row_number, "scenario name"]}')
        print(f'priority_correction_prob: {priority_correction_prob}')
        # 2016 iters, for 2 weeks
        iters = 2016

        # set up dispatch points
        dispatch_pts = set_up_dispatch_pts(dispatch_pts_outline, n_amb, n_ptv, n_sp, n_ip)

        # run model
        n_reps = 20

        patient_output_appended = pd.DataFrame()
        system_output_appended = pd.DataFrame()
        for _ in range(n_reps):
            # set seed
            seed = np.random.randint(0, 1000000)
            patient_outputs, system_outputs = run_sim(
                                    iters=int(iters), 
                                    dispatch_pts=dispatch_pts, 
                                    rates_p1=rates_p1, rates_p2=rates_p2, rates_ppt=rates_ppt,
                                    ptv_travel_scaling=ptv_travel_scaling,
                                    amb_travel_scaling=amb_travel_scaling,
                                    prob_amb=prob_amb,
                                    prob_ip=prob_ip,
                                    ft_communication_prob=ft_communication_prob,
                                    prioritise_p1_prob=prioritise_p1_prob,
                                    prioritise_p2_prob=prioritise_p2_prob,
                                    time_buffers=time_buffers,
                                    mean_wait_times=mean_wait_times,
                                    outputs=['patient', 'system'],
                                    prob_single_staff=prob_single_staff,
                                    modelled_ft=True,
                                    verbose=False,
                                    pb=True,
                                    priority_correction_prob=priority_correction_prob,
                                    restricted_vehicle_movement=False,
                                    reroute_prob=reroute_prob)
            # in patient_outputs rename wait_time to patient_wait_time
            patient_output_appended = pd.concat([patient_output_appended, patient_outputs])
            system_output_appended = pd.concat([system_output_appended, system_outputs])
        #vehicle_locations_appended = pd.concat([vehicle_locations_appended, vehicle_locations])
        # count the number of non-numeric values in patient_wait_time 
        # calculate average response times for each patient type
        
        # save output into csv in folder output_data
        patient_output_appended.to_csv('/Users/skycope/Documents/Thesis/Python/project/output_data/patient_outputs.csv')
        system_output_appended.to_csv('/Users/skycope/Documents/Thesis/Python/project/output_data/system_outputs.csv')
        #vehicle_locations_appended.to_csv('/Users/skycope/Documents/Thesis/Python/project/output_data/vehicle_locations.csv')
        # group by every 144 rows, calculate median RT. lower 0.05 quantile:
        
        # send to R
        patient_output_appended.to_csv('/Users/skycope/Documents/Thesis/R/patient_outputs.csv')
        system_output_appended.to_csv('/Users/skycope/Documents/Thesis/R/system_outputs.csv')


if __name__ == '__main__':
    main()  
