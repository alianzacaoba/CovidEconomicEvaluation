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
c_beta_inf = np.ones(6)*0.0000000000001
c_beta_base = np.ones(6)*0.1
c_beta_sup = np.ones(6)*0.1
c_death_inf = np.zeros(6)
c_death_base = np.ones(6)
c_death_sup = np.ones(6)*2.2
c_arrival_inf = np.ones(6)*1.0
c_arrival_base = np.ones(6)*15
c_arrival_sup = np.ones(6)*30
c_spc_inf = 0
c_spc_base = 0.8
c_spc_sup = 1
calibration_model = Calibration()
c_beta_ant = 0.0
c_death_ant = 0.0
n_iteration = 0
days_deaths = {1: 62, 2: 105, 3: 47, 4: 111, 5: 58, 6: 84}
days_cases = {1: 23, 2: 20, 3: 18, 4: 30, 5: 22, 6: 55}
n_cases = 5*(len(c_beta_base) + len(c_death_base) + len(c_arrival_base)+1)+1
c_total = True
n_changed = 0
previous_error = [100000000000000.0]
while n_changed < 5:
    n_iteration += 1
    print('Cycle number:', n_iteration)
    calibration_model.run_calibration(initial_cases=n_cases, beta_range=[c_beta_inf, c_beta_base, c_beta_sup],
                                      death_range=[c_death_inf, c_death_base, c_death_sup],
                                      arrival_range=[c_arrival_inf, c_arrival_base, c_arrival_sup],
                                      symptomatic_probability_range=[c_spc_inf, c_spc_base, c_spc_sup],
                                      dates={'days_cases': days_cases, 'days_deaths': days_deaths}, total=c_total,
                                      iteration=n_iteration, max_no_improvement=50,
                                      min_value_to_iterate=100000, error_precision=7)
    print(' End of cycle: ', n_iteration)
    print(' New error:', calibration_model.ideal_values['error'])
    print(' Previous errors:', previous_error)
    print(' Improvement: ', float(previous_error[len(previous_error)-1] -
                                  calibration_model.ideal_values['error']))
    print(' No changes in: ', n_changed)
    if float(calibration_model.ideal_values['error']) < float(previous_error[len(previous_error)-1]):
        c_beta_base = np.array(calibration_model.ideal_values['beta'])
        c_death_base = np.array(calibration_model.ideal_values['dc'])
        c_arrival_base = np.array(calibration_model.ideal_values['arrival'])
        c_spc_base = min(max(0.00000000000001, float(calibration_model.ideal_values['spc'])), 1.0)
        radius = np.maximum(np.absolute(c_beta_base - c_beta_inf), np.absolute(c_beta_base - c_beta_sup))/2
        c_beta_inf = np.maximum(np.minimum(c_beta_inf, c_beta_base), c_beta_base - radius)
        c_beta_sup = np.minimum(np.maximum(c_beta_sup, c_beta_base), c_beta_base + radius)
        radius = np.maximum(np.absolute(c_death_base - c_death_inf), np.absolute(c_death_base - c_death_sup))/2
        c_death_inf = np.maximum(np.minimum(c_death_inf, c_death_base), c_death_base - radius)
        c_death_sup = np.minimum(np.maximum(c_death_sup, c_death_base), c_death_base + radius)
        radius = np.maximum(np.absolute(c_arrival_base - c_arrival_inf), np.absolute(c_arrival_base - c_arrival_sup))/2
        c_arrival_inf = np.maximum(np.minimum(c_arrival_inf, c_arrival_base), c_arrival_base - radius)
        c_arrival_sup = np.minimum(np.maximum(c_arrival_sup, c_arrival_base), c_arrival_base + radius)
        radius = max(abs(c_spc_base - c_spc_inf), abs(c_spc_base - c_spc_sup))/2
        c_spc_inf = max(min(c_spc_base, c_spc_inf), c_spc_base - radius)
        c_spc_sup = min(max(c_spc_base, c_spc_sup), c_spc_base + radius)
    if round(float(calibration_model.ideal_values['error']), 7) < round(float(previous_error[len(previous_error) - 1]), 7):
        n_changed = 0
    else:
        n_changed += 1
    previous_error.append(float(calibration_model.ideal_values['error']))

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
with open(DIR_OUTPUT + 'calibration_consolidated_results_from_start_' +
          ('total' if c_total else 'new') + '.json', 'w') as fp:
    json.dump(calibration_model.results, fp)
results_pd_c = pd.DataFrame(calibration_model.results)
c_values = [results_pd_c.columns] + list(results_pd_c.values)
c_wb = Workbook()
c_wb.new_sheet('All_values', data=c_values)
c_wb.save(DIR_OUTPUT + 'calibration_nm_results_from_start_' + ('total' if c_total else 'new') + '.xlsx')
print('Excel ', DIR_OUTPUT + 'calibration_nm_results_from_start_' + ('total' if c_total else 'new') + '.xlsx',
      'exported')
