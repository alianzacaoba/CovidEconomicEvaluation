from pyexcelerate import Workbook
from logic.calibration_nelder_mead import Calibration
import pandas as pd
import numpy as np
import datetime
import json
import time
from root import DIR_OUTPUT

start_processing_s_t = time.process_time()
start_time_t = datetime.datetime.now()

''' beta : (0.015214414449999901, 0.014687559412714796, 0.013534606893713143, 0.014790915789624548, 
0.013498338154962937, 0.012507595349096866)
dc : (1.4617384461017835, 0.9368353193002283, 1.1204263956170062, 1.0444619838337001, 1.4319438462943168, 
1.6058498865338446)
arrival : (27.183292359067348, 9.730573765183337, 33.90194771637668, 13.544846736198476, 33.80045229696106, 
34.6074539432445)
error_cases : (0.3330180059536827, 0.13823762851975097, 0.1929584303547453, 0.1634628364922416, 0.2167009591928125, 
0.14154887323094312)
error_deaths : (0.3513654012590485, 0.19525489509516827, 0.1778453094031537, 0.22200372950693953, 0.25795356231270633, 
0.17969900866341648)
error : 0.2031598770822587
'''

c_beta_base = np.array((0.015214414449999901, 0.014687559412714796, 0.013534606893713143, 0.014790915789624548,
                        0.013498338154962937, 0.012507595349096866))
c_beta_inf = c_beta_base*0.9
c_beta_sup = c_beta_base*1.1
c_death_base = np.array((1.4617384461017835, 0.9368353193002283, 1.1204263956170062, 1.0444619838337001,
                         1.4319438462943168, 1.6058498865338446))
c_death_inf = c_death_base*.9
c_death_sup = c_death_base*1.1
c_arrival_inf = np.ones(6)*1.0
c_arrival_base = np.array((27.183292359067348, 9.730573765183337, 33.90194771637668, 13.544846736198476,
                           33.80045229696106, 34.6074539432445))
c_arrival_sup = np.ones(6)*36

calibration_model = Calibration()
c_beta_ant = 0.0
c_death_ant = 0.0
n_iteration = 0
days_deaths = {1: 62, 2: 105, 3: 47, 4: 111, 5: 58, 6: 84}
days_cases = {1: 23, 2: 20, 3: 18, 4: 30, 5: 22, 6: 55}
n_cases = 5*(len(c_beta_base) + len(c_death_base) + len(c_arrival_base))+1
c_total = True
n_changed = 0
previous_error = 100000000000000
while n_changed < 2:
    n_iteration += 1
    print('Cycle number:', n_iteration)
    calibration_model.run_calibration(initial_cases=n_cases, beta_range=[c_beta_inf, c_beta_base, c_beta_sup],
                                      death_range=[c_death_inf, c_death_base, c_death_sup],
                                      arrival_range=[c_arrival_inf, c_arrival_base, c_arrival_sup],
                                      dates={'days_cases': days_cases, 'days_deaths': days_deaths}, total=c_total,
                                      iteration=10000+n_iteration, max_shrinks=5, max_no_improvement=50,
                                      min_value_to_iterate=1000)
    if calibration_model.current_results[0]['error'] < previous_error:
        n_changed = 0
        previous_error = calibration_model.ideal_values['error']
        c_beta_base = np.array(calibration_model.ideal_values['beta'])
        c_death_base = np.array(calibration_model.ideal_values['dc'])
        c_arrival_base = np.array(calibration_model.ideal_values['arrival'])
        c_beta_inf = np.minimum(np.average([c_beta_base, c_beta_inf], axis=0), c_beta_base * 0.9)
        c_beta_sup = np.maximum(np.average([c_beta_base, c_beta_sup], axis=0), c_beta_base * 1.1)
        c_death_inf = np.minimum(np.average([c_death_base, c_death_inf], axis=0), c_death_base * 0.9)
        c_death_sup = np.maximum(np.average([c_death_base, c_death_sup], axis=0), c_death_sup * 1.1)
        c_arrival_inf = np.minimum(np.average([c_arrival_base, c_arrival_inf], axis=0), c_arrival_base * 0.9)
        c_arrival_sup = np.maximum(np.average([c_arrival_base, c_arrival_sup], axis=0), c_arrival_base * 1.1)
    else:
        n_changed += 1

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
with open(DIR_OUTPUT + 'calibration_consolidated_results_5_' + ('total' if c_total else 'new') + '.json', 'w') as fp:
    json.dump(calibration_model.results, fp)
results_pd_c = pd.DataFrame(calibration_model.results)
c_values = [results_pd_c.columns] + list(results_pd_c.values)
c_wb = Workbook()
c_wb.new_sheet('All_values', data=c_values)
c_wb.save(DIR_OUTPUT + 'calibration_consolidated_results_5_' + ('total' if c_total else 'new') + '.xlsx')
print('Excel ', DIR_OUTPUT + 'calibration_consolidated_results_5_' + ('total' if c_total else 'new') + '.xlsx',
      'exported')
