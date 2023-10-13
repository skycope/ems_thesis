import pandas as pd
import numpy as np
import string
import random
import os

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
    print(f'running sensitivity analysis (second time) with run id {run_id}...')
    # print working directory
    # read in sensitivity_parameters
    sensitivity_parameters = pd.read_csv('lhs_parameter_sets.csv')
    sensitivity_outputs = pd.DataFrame({
    })
    # this is a for loop iterating over the df
    # sample 40 rows to run on the chosen core
    sampled_parameters = sensitivity_parameters.sample(30)
    # reset indices
    sampled_parameters = sampled_parameters.reset_index(drop=True)
    for row_number in sampled_parameters.index:
        # extract parameter values form sensitivity_parameters
        n_amb = sampled_parameters.loc[row_number, 'n_amb'].replace("'", '"')
        n_ptv = sampled_parameters.loc[row_number, 'n_ptv'].replace("'", '"')
        n_sp = sampled_parameters.loc[row_number, 'n_sp'].replace("'", '"')
        n_ip = sampled_parameters.loc[row_number, 'n_ip'].replace("'", '"')
        ptv_travel_scaling = 1
        amb_travel_scaling = 1
        prob_amb = sampled_parameters.loc[row_number, 'prob_amb'].replace("'", '"')
        prob_ip = sampled_parameters.loc[row_number, 'prob_ip'].replace("'", '"')
        time_buffers = sampled_parameters.loc[row_number, 'time_buffers'].replace("'", '"')
        mean_wait_times = sampled_parameters.loc[row_number, 'mean_wait_times'].replace("'", '"')
        prob_single_staff = sampled_parameters.loc[row_number, 'prob_single_staff'].replace("'", '"')
        ft_communication_prob = sampled_parameters.loc[row_number, 'ft_communication_prob']
        prioritise_p1_prob = sampled_parameters.loc[row_number, 'prioritise_p1_prob']
        prioritise_p2_prob = sampled_parameters.loc[row_number, 'prioritise_p2_prob']
        reroute_prob = sampled_parameters.loc[row_number, 'reroute_prob']
        priority_correction_prob = sampled_parameters.loc[row_number, 'priority_correction_prob']
        # convert to bool (it's FALSE or TRUE)
        restricted_vehicle_movement = False
        # print out parameters
        # 2016 iters, for 2 weeks
        iters = 2016

        # set up dispatch points
        dispatch_pts = set_up_dispatch_pts(dispatch_pts_outline, n_amb, n_ptv, n_sp, n_ip)

        # run model
        n_reps = 1

        seeds = []
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
            seeds = seeds + [seed]

            # add a column called weekend
            patient_outputs['weekend'] = patient_outputs['tick'].apply(weekend_weekday)
            # concat to it
            sensitivity_outputs = sensitivity_outputs.reset_index(drop=True)

            sensitivity_outputs = pd.concat([sensitivity_outputs, pd.DataFrame({
                'seed': seeds[-1],
                # average response times
                'avg_rt_p1': patient_outputs.loc[patient_outputs.patient_type == 'p1'].patient_wait_time.mean(),
                'avg_rt_p2': patient_outputs.loc[patient_outputs.patient_type == 'p2'].patient_wait_time.mean(),
                'avg_rt_ppt': patient_outputs.loc[patient_outputs.patient_type == 'ppt'].patient_wait_time.mean(),
                # median response times
                'median_rt_p1': patient_outputs.loc[patient_outputs.patient_type == 'p1'].patient_wait_time.median(),
                'median_rt_p2': patient_outputs.loc[patient_outputs.patient_type == 'p2'].patient_wait_time.median(),
                'median_rt_ppt': patient_outputs.loc[patient_outputs.patient_type == 'ppt'].patient_wait_time.median(),
                # 95% quantiles
                'lower_95_rt_p1': patient_outputs.loc[patient_outputs.patient_type == 'p1'].patient_wait_time.quantile(0.025),
                'lower_95_rt_p2': patient_outputs.loc[patient_outputs.patient_type == 'p2'].patient_wait_time.quantile(0.025),
                'lower_95_rt_ppt': patient_outputs.loc[patient_outputs.patient_type == 'ppt'].patient_wait_time.quantile(0.025),
                'upper_95_rt_p1': patient_outputs.loc[patient_outputs.patient_type == 'p1'].patient_wait_time.quantile(0.975),
                'upper_95_rt_p2': patient_outputs.loc[patient_outputs.patient_type == 'p2'].patient_wait_time.quantile(0.975),
                'upper_95_rt_ppt': patient_outputs.loc[patient_outputs.patient_type == 'ppt'].patient_wait_time.quantile(0.975),            
                # triage classification red
                'median_rt_red': patient_outputs.loc[patient_outputs.triage_classification == 'red'].patient_wait_time.median(),
                'lower_95_rt_red': patient_outputs.loc[patient_outputs.triage_classification == 'red'].patient_wait_time.quantile(0.025),
                'upper_95_rt_red': patient_outputs.loc[patient_outputs.triage_classification == 'red'].patient_wait_time.quantile(0.975),
                # triage classification yellow
                'median_rt_yellow': patient_outputs.loc[patient_outputs.triage_classification == 'yellow'].patient_wait_time.median(),
                'lower_95_rt_yellow': patient_outputs.loc[patient_outputs.triage_classification == 'yellow'].patient_wait_time.quantile(0.025),
                'upper_95_rt_yellow': patient_outputs.loc[patient_outputs.triage_classification == 'yellow'].patient_wait_time.quantile(0.975),
                # triage classification green
                'median_rt_green': patient_outputs.loc[patient_outputs.triage_classification == 'green'].patient_wait_time.median(),
                'lower_95_rt_green': patient_outputs.loc[patient_outputs.triage_classification == 'green'].patient_wait_time.quantile(0.025),
                'upper_95_rt_green': patient_outputs.loc[patient_outputs.triage_classification == 'green'].patient_wait_time.quantile(0.975),
                # now p1 RT for each region
                'median_rt_p1_region_5': patient_outputs.loc[(patient_outputs.patient_type == 'p1') & (patient_outputs.closest_central_ward == '5')].patient_wait_time.median(),
                'lower_95_rt_p1_region_5': patient_outputs.loc[(patient_outputs.patient_type == 'p1') & (patient_outputs.closest_central_ward == '5')].patient_wait_time.quantile(0.025),
                'upper_95_rt_p1_region_5': patient_outputs.loc[(patient_outputs.patient_type == 'p1') & (patient_outputs.closest_central_ward == '5')].patient_wait_time.quantile(0.975),
                # and for region 25
                'median_rt_p1_region_25': patient_outputs.loc[(patient_outputs.patient_type == 'p1') & (patient_outputs.closest_central_ward == '25')].patient_wait_time.median(),
                'lower_95_p1_region_25': patient_outputs.loc[(patient_outputs.patient_type == 'p1') & (patient_outputs.closest_central_ward == '25')].patient_wait_time.quantile(0.025),
                'upper_95_p1_region_25': patient_outputs.loc[(patient_outputs.patient_type == 'p1') & (patient_outputs.closest_central_ward == '25')].patient_wait_time.quantile(0.975),
                # and for region 51
                'median_rt_p1_region_51': patient_outputs.loc[(patient_outputs.patient_type == 'p1') & (patient_outputs.closest_central_ward == '51')].patient_wait_time.median(),
                'lower_95_p1_region_51': patient_outputs.loc[(patient_outputs.patient_type == 'p1') & (patient_outputs.closest_central_ward == '51')].patient_wait_time.quantile(0.025),
                'upper_95_p1_region_51': patient_outputs.loc[(patient_outputs.patient_type == 'p1') & (patient_outputs.closest_central_ward == '51')].patient_wait_time.quantile(0.975),
                # and for region 58
                'median_rt_p1_region_58': patient_outputs.loc[(patient_outputs.patient_type == 'p1') & (patient_outputs.closest_central_ward == '58')].patient_wait_time.median(),
                'lower_95_p1_region_58': patient_outputs.loc[(patient_outputs.patient_type == 'p1') & (patient_outputs.closest_central_ward == '58')].patient_wait_time.quantile(0.025),
                'upper_95_p1_region_58': patient_outputs.loc[(patient_outputs.patient_type == 'p1') & (patient_outputs.closest_central_ward == '58')].patient_wait_time.quantile(0.975),
                # and for region 34
                'median_rt_p1_region_34': patient_outputs.loc[(patient_outputs.patient_type == 'p1') & (patient_outputs.closest_central_ward == '34')].patient_wait_time.median(),
                'lower_95_p1_region_34': patient_outputs.loc[(patient_outputs.patient_type == 'p1') & (patient_outputs.closest_central_ward == '34')].patient_wait_time.quantile(0.025),
                'upper_95_p1_region_34': patient_outputs.loc[(patient_outputs.patient_type == 'p1') & (patient_outputs.closest_central_ward == '34')].patient_wait_time.quantile(0.975),
                # extract values from system_outputs
                'avg_amb_idle': system_outputs.amb_idle.mean(),
                'avg_amb_enroute': system_outputs.amb_enroute.mean(),
                'avg_amb_occupied': system_outputs.amb_occupied.mean(),
                'avg_ptv_idle': system_outputs.ptv_idle.mean(),
                'avg_ptv_enroute': system_outputs.ptv_enroute.mean(),
                'avg_ptv_occupied': system_outputs.ptv_occupied.mean(),
                'avg_ip_idle': system_outputs.ip_idle.mean(),
                'avg_ip_enroute': system_outputs.ip_enroute.mean(),
                'avg_sp_idle': system_outputs.sp_idle.mean(),
                'avg_sp_enroute': system_outputs.sp_enroute.mean(),
                # now we extract prop_two_ip_p1, prop_two_ip_p2, prop_two_ip_ppt
                'prop_two_ip_p1': patient_outputs.loc[(patient_outputs.patient_type == 'p1') & (patient_outputs.staff_1_type == 'ip') & (patient_outputs.staff_2_type == 'ip')].shape[0]/patient_outputs.loc[(patient_outputs.patient_type == 'p1')].shape[0],
                'prop_two_ip_p2': patient_outputs.loc[(patient_outputs.patient_type == 'p2') & (patient_outputs.staff_1_type == 'ip') & (patient_outputs.staff_2_type == 'ip')].shape[0]/patient_outputs.loc[(patient_outputs.patient_type == 'p2')].shape[0],
                'prop_two_ip_ppt': patient_outputs.loc[(patient_outputs.patient_type == 'ppt') & (patient_outputs.staff_1_type == 'ip') & (patient_outputs.staff_2_type == 'ip')].shape[0]/patient_outputs.loc[(patient_outputs.patient_type == 'ppt')].shape[0],
                # now we extract prop_two_sp_p1, prop_two_sp_p2, prop_two_sp_ppt
                'prop_two_sp_p1': patient_outputs.loc[(patient_outputs.patient_type == 'p1') & (patient_outputs.staff_1_type == 'sp') & (patient_outputs.staff_2_type == 'sp')].shape[0]/patient_outputs.loc[(patient_outputs.patient_type == 'p1')].shape[0],
                'prop_two_sp_p2': patient_outputs.loc[(patient_outputs.patient_type == 'p2') & (patient_outputs.staff_1_type == 'sp') & (patient_outputs.staff_2_type == 'sp')].shape[0]/patient_outputs.loc[(patient_outputs.patient_type == 'p2')].shape[0],
                'prop_two_sp_ppt': patient_outputs.loc[(patient_outputs.patient_type == 'ppt') & (patient_outputs.staff_1_type == 'sp') & (patient_outputs.staff_2_type == 'sp')].shape[0]/patient_outputs.loc[(patient_outputs.patient_type == 'ppt')].shape[0],
                # now we extract prop_both_p1, prop_both_p2, prop_both_ppt (both kinds of staff members, ip and sp)
                # staff_1_type is ip, staff_2_type is sp OR staff_1_type is sp, staff_2_type is ip
                'prop_both_p1': patient_outputs.loc[(patient_outputs.patient_type == 'p1') & 
                                                    ((patient_outputs.staff_1_type == 'ip') & 
                                                     (patient_outputs.staff_2_type == 'sp') | (patient_outputs.staff_1_type == 'sp') & (patient_outputs.staff_2_type == 'ip')
                                                     )].shape[0]/patient_outputs.loc[(patient_outputs.patient_type == 'p1')].shape[0],
                'prop_both_p2': patient_outputs.loc[(patient_outputs.patient_type == 'p2') &
                                                    ((patient_outputs.staff_1_type == 'ip') &
                                                        (patient_outputs.staff_2_type == 'sp') | (patient_outputs.staff_1_type == 'sp') & (patient_outputs.staff_2_type == 'ip')
                                                        )].shape[0]/patient_outputs.loc[(patient_outputs.patient_type == 'p2')].shape[0],
                'prop_both_ppt': patient_outputs.loc[(patient_outputs.patient_type == 'ppt') &
                                                    ((patient_outputs.staff_1_type == 'ip') &
                                                        (patient_outputs.staff_2_type == 'sp') | (patient_outputs.staff_1_type == 'sp') & (patient_outputs.staff_2_type == 'ip')
                                                        )].shape[0]/patient_outputs.loc[(patient_outputs.patient_type == 'ppt')].shape[0],
                # now we extract prop_one_sp_p1, prop_one_sp_p2, prop_one_sp_ppt
                # staff_1_type is sp, staff_2_type is '' or staff_1_type is '', staff_2_type is sp
              'prop_one_sp_p1': patient_outputs.loc[
                    (patient_outputs.patient_type == 'p1') & 
                    (((patient_outputs.staff_1_type == 'sp') & (~patient_outputs.staff_2_type.isin(['ip', 'sp']))) | 
                    ((~patient_outputs.staff_1_type.isin(['ip', 'sp'])) & (patient_outputs.staff_2_type == 'sp')))
                ].shape[0] / patient_outputs.loc[(patient_outputs.patient_type == 'p1')].shape[0],

             'prop_one_sp_p2': patient_outputs.loc[
                    (patient_outputs.patient_type == 'p2') & 
                    (((patient_outputs.staff_1_type == 'sp') & (~patient_outputs.staff_2_type.isin(['ip', 'sp']))) | 
                    ((~patient_outputs.staff_1_type.isin(['ip', 'sp'])) & (patient_outputs.staff_2_type == 'sp')))
                ].shape[0] / patient_outputs.loc[(patient_outputs.patient_type == 'p2')].shape[0],

            'prop_one_sp_ppt': patient_outputs.loc[
                    (patient_outputs.patient_type == 'ppt') & 
                    (((patient_outputs.staff_1_type == 'sp') & (~patient_outputs.staff_2_type.isin(['ip', 'sp']))) | 
                    ((~patient_outputs.staff_1_type.isin(['ip', 'sp'])) & (patient_outputs.staff_2_type == 'sp')))
                ].shape[0] / patient_outputs.loc[(patient_outputs.patient_type == 'ppt')].shape[0],

            'prop_one_ip_p1': patient_outputs.loc[
                    (patient_outputs.patient_type == 'p1') & 
                    (((patient_outputs.staff_1_type == 'ip') & (~patient_outputs.staff_2_type.isin(['ip', 'sp']))) | 
                    ((~patient_outputs.staff_1_type.isin(['ip', 'sp'])) & (patient_outputs.staff_2_type == 'ip')))
                ].shape[0] / patient_outputs.loc[(patient_outputs.patient_type == 'p1')].shape[0],


            'prop_one_ip_p2': patient_outputs.loc[
                                (patient_outputs.patient_type == 'p2') & 
                                (((patient_outputs.staff_1_type == 'ip') & (~patient_outputs.staff_2_type.isin(['ip', 'sp']))) |
                                ((~patient_outputs.staff_1_type.isin(['ip', 'sp'])) & (patient_outputs.staff_2_type == 'ip')))
                            ].shape[0] / patient_outputs.loc[(patient_outputs.patient_type == 'p2')].shape[0],

            'prop_one_ip_ppt': patient_outputs.loc[
                                (patient_outputs.patient_type == 'ppt') & 
                                (((patient_outputs.staff_1_type == 'ip') & (~patient_outputs.staff_2_type.isin(['ip', 'sp']))) |
                                ((~patient_outputs.staff_1_type.isin(['ip', 'sp'])) & (patient_outputs.staff_2_type == 'ip')))
                            ].shape[0] / patient_outputs.loc[(patient_outputs.patient_type == 'ppt')].shape[0],
                # number calls answered, by vehicle type, call type
                'n_p1_amb': len(patient_outputs.loc[(patient_outputs.patient_type == 'p1') & (patient_outputs.vehicle_type == 'amb')]),
                'n_p2_amb': len(patient_outputs.loc[(patient_outputs.patient_type == 'p2') & (patient_outputs.vehicle_type == 'amb')]),
                'n_ppt_amb': len(patient_outputs.loc[(patient_outputs.patient_type == 'ppt') & (patient_outputs.vehicle_type == 'amb')]),
                'n_p1_ptv': len(patient_outputs.loc[(patient_outputs.patient_type == 'p1') & (patient_outputs.vehicle_type == 'ptv')]),
                'n_p2_ptv': len(patient_outputs.loc[(patient_outputs.patient_type == 'p2') & (patient_outputs.vehicle_type == 'ptv')]),
                'n_ppt_ptv': len(patient_outputs.loc[(patient_outputs.patient_type == 'ppt') & (patient_outputs.vehicle_type == 'ptv')]),

                # number of calls completed by vehicle type, call type
                'n_p1_amb_completed': len(patient_outputs.loc[(patient_outputs.patient_type == 'p1') & (patient_outputs.vehicle_type == 'amb') & (patient_outputs.ft == '')]),
                'n_p2_amb_completed': len(patient_outputs.loc[(patient_outputs.patient_type == 'p2') & (patient_outputs.vehicle_type == 'amb') & (patient_outputs.ft == '')]),
                'n_ppt_amb_completed': len(patient_outputs.loc[(patient_outputs.patient_type == 'ppt') & (patient_outputs.vehicle_type == 'amb') & (patient_outputs.ft == '')]),
                'n_p1_ptv_completed': len(patient_outputs.loc[(patient_outputs.patient_type == 'p1') & (patient_outputs.vehicle_type == 'ptv') & (patient_outputs.ft == '')]),
                'n_p2_ptv_completed': len(patient_outputs.loc[(patient_outputs.patient_type == 'p2') & (patient_outputs.vehicle_type == 'ptv') & (patient_outputs.ft == '')]),
                'n_ppt_ptv_completed': len(patient_outputs.loc[(patient_outputs.patient_type == 'ppt') & (patient_outputs.vehicle_type == 'ptv') & (patient_outputs.ft == '')]),
                
                # break down RT by weekend, weekday
                'median_rt_p1_weekday': patient_outputs.loc[(patient_outputs.patient_type == 'p1') & (patient_outputs.weekend == 'weekday')].patient_wait_time.median(),
                'median_rt_p1_weekend': patient_outputs.loc[(patient_outputs.patient_type == 'p1') & (patient_outputs.weekend == 'weekend')].patient_wait_time.median(),
                'median_rt_p2_weekday': patient_outputs.loc[(patient_outputs.patient_type == 'p2') & (patient_outputs.weekend == 'weekday')].patient_wait_time.median(),
                'median_rt_p2_weekend': patient_outputs.loc[(patient_outputs.patient_type == 'p2') & (patient_outputs.weekend == 'weekend')].patient_wait_time.median(),
                'median_rt_ppt_weekday': patient_outputs.loc[(patient_outputs.patient_type == 'ppt') & (patient_outputs.weekend == 'weekday')].patient_wait_time.median(),
                'median_rt_ppt_weekend': patient_outputs.loc[(patient_outputs.patient_type == 'ppt') & (patient_outputs.weekend == 'weekend')].patient_wait_time.median(),
                # and 95% quantiles
                'lower_95_rt_p1_weekday': patient_outputs.loc[(patient_outputs.patient_type == 'p1') & (patient_outputs.weekend == 'weekday')].patient_wait_time.quantile(0.025),
                'lower_95_rt_p1_weekend': patient_outputs.loc[(patient_outputs.patient_type == 'p1') & (patient_outputs.weekend == 'weekend')].patient_wait_time.quantile(0.025),
                'lower_95_rt_p2_weekday': patient_outputs.loc[(patient_outputs.patient_type == 'p2') & (patient_outputs.weekend == 'weekday')].patient_wait_time.quantile(0.025),
                'lower_95_rt_p2_weekend': patient_outputs.loc[(patient_outputs.patient_type == 'p2') & (patient_outputs.weekend == 'weekend')].patient_wait_time.quantile(0.025),
                'lower_95_rt_ppt_weekday': patient_outputs.loc[(patient_outputs.patient_type == 'ppt') & (patient_outputs.weekend == 'weekday')].patient_wait_time.quantile(0.025),
                'lower_95_rt_ppt_weekend': patient_outputs.loc[(patient_outputs.patient_type == 'ppt') & (patient_outputs.weekend == 'weekend')].patient_wait_time.quantile(0.025),
                'upper_95_rt_p1_weekday': patient_outputs.loc[(patient_outputs.patient_type == 'p1') & (patient_outputs.weekend == 'weekday')].patient_wait_time.quantile(0.975),
                'upper_95_rt_p1_weekend': patient_outputs.loc[(patient_outputs.patient_type == 'p1') & (patient_outputs.weekend == 'weekend')].patient_wait_time.quantile(0.975),
                'upper_95_rt_p2_weekday': patient_outputs.loc[(patient_outputs.patient_type == 'p2') & (patient_outputs.weekend == 'weekday')].patient_wait_time.quantile(0.975),
                'upper_95_rt_p2_weekend': patient_outputs.loc[(patient_outputs.patient_type == 'p2') & (patient_outputs.weekend == 'weekend')].patient_wait_time.quantile(0.975),
                'upper_95_rt_ppt_weekday': patient_outputs.loc[(patient_outputs.patient_type == 'ppt') & (patient_outputs.weekend == 'weekday')].patient_wait_time.quantile(0.975),
                'upper_95_rt_ppt_weekend': patient_outputs.loc[(patient_outputs.patient_type == 'ppt') & (patient_outputs.weekend == 'weekend')].patient_wait_time.quantile(0.975),
            }, index=[0])], ignore_index=True)
    sensitivity_outputs = sensitivity_outputs.reset_index(drop=True)
    sensitivity_outputs = pd.concat([sampled_parameters, sensitivity_outputs], axis=1)
    # write to csv
    sensitivity_outputs.to_csv(f'making_sens_{run_id}.csv', index=False)

if __name__ == '__main__':
    main()
