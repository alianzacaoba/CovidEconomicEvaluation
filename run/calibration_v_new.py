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

c_beta_inf = np.zeros(6)
c_beta_base = (0.016531517382506206, 0.015117576752108406, 0.014833108642790863, 0.015173372335523787,
               0.014551978794636808, 0.013349006162455464)
c_beta_sup = np.ones(6)*0.1
c_death_inf = np.ones(6)
c_death_base = (1.1922174346854706, 0.7938417000314903, 1.1184542996783413, 1.1556253539670713, 0.976135469790154,
                1.1722890049556356)
c_death_sup = np.ones(6)*2.2
c_arrival_inf = np.ones(6)*1.0
c_arrival_base = (18.797792184172703, 14.437873233137548, 19.478914801206944, 17.936837399957028, 19.595558955321536,
                  20.96301233051736)
c_arrival_sup = np.ones(6)*30
calibration_model = Calibration()
c_beta_ant = 0.0
c_death_ant = 0.0
n_iteration = 0
days_deaths = {1: 59, 2: 119, 3: 82, 4: 127, 5: 48, 6: 159}
days_cases = {1: 17, 2: 10, 3: 12, 4: 19, 5: 15, 6: 49}
n_cases = 5*(len(c_beta_base) + len(c_death_base) + len(c_arrival_base))+1
c_total = False
n_changed = 0
previous_error = 100000000000000
while n_changed < 2:
    n_iteration += 1
    print('Cycle number:', n_iteration)
    calibration_model.run_calibration(initial_cases=n_cases, beta_range=[c_beta_inf, c_beta_base, c_beta_sup],
                                      death_range=[c_death_inf, c_death_base, c_death_sup],
                                      arrival_range=[c_arrival_inf, c_arrival_base, c_arrival_sup],
                                      dates={'days_cases': days_cases, 'days_deaths': days_deaths}, total=c_total,
                                      iteration=1000+n_iteration, max_shrinks=5, max_no_improvement=50,
                                      min_value_to_iterate=1000)
    if calibration_model.current_results[0]['error'] < previous_error:
        n_changed = 0
        previous_error = calibration_model.ideal_values['error']
        c_beta_base = calibration_model.ideal_values['beta']
        c_death_base = calibration_model.ideal_values['dc']
        c_arrival_base = calibration_model.ideal_values['arrival']
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
with open(DIR_OUTPUT + 'calibration_consolidated_results_2_' + ('total' if c_total else 'new') + '.json', 'w') as fp:
    json.dump(calibration_model.results, fp)
results_pd_c = pd.DataFrame(calibration_model.results)
c_values = [results_pd_c.columns] + list(results_pd_c.values)
c_wb = Workbook()
c_wb.new_sheet('All_values', data=c_values)
c_wb.save(DIR_OUTPUT + 'calibration_nm_results_' + ('total' if c_total else 'new') + '.xlsx')
print('Excel ', DIR_OUTPUT + 'calibration_consolidated_results_' + ('total' if c_total else 'new') + '.xlsx',
      'exported')
