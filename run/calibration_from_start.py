from pyexcelerate import Workbook
from logic.calibration_nelder_mead import Calibration
import pandas as pd
import numpy as np
import datetime
import json
import time
from logic.main_run import MainRun
from root import DIR_OUTPUT
import threading
from threading import Thread, Lock


'''Current best results:
beta : (0.0020075041540949497, 0.0003441234590969293, 0.0011003086287036108, 0.0005610201180875692, 0.0007416826042882902, 0.00022602994458867934)
dc : (0.05104496972532489, 0.028090496301170036, 0.07387331054455204, 0.01482706610568949, 0.18020901002463557, 0.00936497809139487)
arrival : (16.206510753360355, 11.777574321263948, 12.46803134334906, 20.082416160197084, 17.236684968286777, 13.65895328754797)
spc : 0.016302880707472173
error_seroprevalence : (0.35044665600754304, 0.31887083225555923, 3.405558997267762e-05, 0.02059046728806627, 0.0400393105950847, 0.30173205307279866)
error_cases : (0.9155840658243439, 0.9834996815466357, 0.9672646900946691, 1.0139303840251108, 0.9422053459052557, 0.5765870882428812)
error_deaths : (0.9854841499236704, 0.9950050400732476, 0.9948011696931099, 0.9974288781279625, 0.9709288061895043, 0.9683642085240344)
error : 0.24485594938411853'''

weights = [(50, 10, 1), (10, 5, 1), (3, 2, 1), (1, 1, 1)]

c_beta_base = np.ones(6)*0.005
c_death_base = np.ones(6)*0.05
c_arrival_base = np.ones(6)*5.0
c_spc_base = 0.9
calibration_model = Calibration()

resulting_values = dict()
error_precision = 4
for weight in weights:
    error_precision += 2
    c_beta_inf = c_beta_base * 0.75
    c_beta_sup = c_beta_base * 1.25
    c_death_inf = c_death_base * 0.75
    c_death_sup = c_death_base * 1.25
    c_arrival_inf = c_arrival_base * 0.75
    c_arrival_sup = c_arrival_base * 1.25
    c_spc_inf = c_spc_base * 0.75
    c_spc_sup = c_spc_base * 1.25
    start_processing_s_t = time.process_time()
    start_time_t = datetime.datetime.now()
    n_iteration = 0
    days_deaths = {1: 62, 2: 105, 3: 47, 4: 111, 5: 58, 6: 84}
    days_cases = {1: 23, 2: 20, 3: 18, 4: 30, 5: 22, 6: 55}
    n_cases = 42
    c_total = True
    n_changed = True
    previous_error = [100000000000000.0]
    model_run = MainRun()
    while n_changed:
        n_iteration += 1
        print('Cycle number:', n_iteration)
        calibration_model.run_calibration(initial_cases=n_cases, beta_range=[c_beta_inf, c_beta_base, c_beta_sup],
                                          death_range=[c_death_inf, c_death_base, c_death_sup],
                                          arrival_range=[c_arrival_inf, c_arrival_base, c_arrival_sup],
                                          symptomatic_probability_range=[c_spc_inf, c_spc_base, c_spc_sup],
                                          dates={'days_cases': days_cases, 'days_deaths': days_deaths},
                                          total=c_total, iteration=n_iteration, max_no_improvement=100,
                                          min_value_to_iterate=100000, error_precision=error_precision, weights=weight)
        print(' End of cycle: ', n_iteration)
        print(' New error:', calibration_model.ideal_values['error'])
        print(' Previous errors:', previous_error)
        print(' Improvement: ', float(previous_error[len(previous_error)-1] -
                                      calibration_model.ideal_values['error']))
        if float(calibration_model.ideal_values['error']) < float(previous_error[len(previous_error)-1]):
            c_beta_base = np.array(calibration_model.ideal_values['beta'])
            c_death_base = np.array(calibration_model.ideal_values['dc'])
            c_arrival_base = np.array(calibration_model.ideal_values['arrival'])
            c_spc_base = min(max(0.00000000000001, float(calibration_model.ideal_values['spc'])), 1.0)
            radius = np.maximum(np.absolute(c_beta_base - c_beta_inf), np.absolute(c_beta_base - c_beta_sup))/2
            c_beta_inf = np.maximum(c_beta_base - radius, np.zeros(6))
            c_beta_sup = np.minimum(c_beta_base + radius, np.ones(6))
            radius = np.maximum(np.absolute(c_death_base - c_death_inf), np.absolute(c_death_base - c_death_sup))/2
            c_death_inf = np.maximum(c_death_base - radius, np.zeros(6))
            c_death_sup = c_death_base + radius
            radius = np.maximum(np.absolute(c_arrival_base - c_arrival_inf), np.absolute(c_arrival_base -
                                                                                         c_arrival_sup))/2
            c_arrival_inf = np.maximum(c_arrival_base - radius, np.zeros(6))
            c_arrival_sup = c_arrival_base + radius
            radius = max(abs(c_spc_base - c_spc_inf), abs(c_spc_base - c_spc_sup))/2
            c_spc_inf = max(c_spc_base - radius, 0.00000000000001)
            c_spc_sup = min(c_spc_base + radius, 1.0)
        if round(float(calibration_model.ideal_values['error']), 6) < round(float(previous_error[len(previous_error) - 1]), 6):
            continue
        else:
            n_changed = False
        previous_error.append(float(calibration_model.ideal_values['error']))
        model_run.run_quality_test(c_beta_base, c_death_base, c_arrival_base, c_spc_base,
                                   'NM_cycle_W_' + str(weights[0]) + '_' + str(weights[1]) + '_' +
                                   str(weights[2]) + '_' + str(n_iteration))
    end_processing_s_t = time.process_time()
    end_time_t = datetime.datetime.now()
    print('Performance: {0}'.format(end_processing_s_t - start_processing_s_t))
    time_diff = (end_time_t - start_time_t)
    execution_time = time_diff.total_seconds()
    mm = int(execution_time / 60)
    ss = int(execution_time % 60)
    print('Total Execution Time: {0} minutes {1} seconds'.format(mm, ss))
    print('Total Execution Time: {0} milliseconds'.format(execution_time * 1000))
    print('Total cycles:', n_iteration)
    print('Optimum:')
    for oc in calibration_model.ideal_values:
        print(" ", oc, ":", calibration_model.ideal_values[oc])
    file_name = DIR_OUTPUT + 'calibration_consolidated_nm_results_' + 'W_' + weights[0] + '_' + weights[1] + '_' + \
                weights[2] + ' ' + ('total' if c_total else 'new')
    with open(file_name + '.json', 'w') as fp:
        json.dump(calibration_model.results, fp)
    results_pd_c = pd.DataFrame(calibration_model.results)
    c_values = [results_pd_c.columns] + list(results_pd_c.values)
    c_wb = Workbook()
    c_wb.new_sheet('All_values', data=c_values)
    c_wb.save(file_name + '.xlsx')
    print('Excel ', file_name + '.xlsx', 'exported')
