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
c_beta_base = [0.017489941199482377, 0.016112349678480548, 0.01596144562497532, 0.01604969472492147,
                0.01594032720527475, 0.014592795166043638]
c_beta_sup = np.ones(6)*0.1
c_death_inf = np.ones(6)
c_death_base = [1.1090872690815294, 0.9191014766070283, 1.1509118259982891, 1.145831661818373, 0.9706014276065191,
                1.0660723105523215]
c_death_sup = np.ones(6)*2.2
c_arrival_inf = np.ones(6)*1.0
c_arrival_base = [10.814665023817005, 5.140364799835279, 13.878754516005142, 8.18035656605186, 14.54898647091474,
                  14.297555543354076]
c_arrival_sup = np.ones(6)*20
calibration_model = Calibration()
c_beta_ant = 0.0
c_death_ant = 0.0
n_iteration = 0
days_deaths = {1: 62, 2: 105, 3: 47, 4: 111, 5: 58, 6: 84}
days_cases = {1: 23, 2: 20, 3: 18, 4: 30, 5: 22, 6: 55}
n_cases = 5*(len(c_beta_base) + len(c_death_base) + len(c_arrival_base))+1
c_total = True
changed = True
previous_error = 100000000000000
'''beta : [0.017489941199482377, 0.016112349678480548, 0.01596144562497532, 0.01604969472492147, 0.01594032720527475, 
0.014592795166043638]
  dc : [1.1090872690815294, 0.9191014766070283, 1.1509118259982891, 1.145831661818373, 0.9706014276065191,   
  1.0660723105523215]
  arrival : [10.814665023817005, 5.140364799835279, 13.878754516005142, 8.18035656605186, 14.54898647091474,  
  14.297555543354076]
  error_cases : [0.32795963992562643, 0.059111583613160545, 0.12873686925978015, 0.06807732778319682,   
  0.19990458959075885, 0.3150203656568336]
  error_deaths : [0.6190063824820542, 0.13475290700208564, 0.13533174063381895, 0.13104625129662278, 0.3147571880798705,  
  0.39963180546622257]
  error : 3.932147026619387'''

'''(0.017489941199482377, 0.015945150490559196, 0.01596144562497532, 0.01604969472492147, 0.01594032720527475, 
0.014592795166043638)
(1.1090872690815294, 0.9565891980793171, 1.1509118259982891, 1.145831661818373, 0.9706014276065191, 1.0660723105523215)
(10.814665023817005, 5.2071519149489225, 13.878754516005142, 8.18035656605186, 14.54898647091474, 14.297555543354076)
(0.23341737686509315, 0.06999851923845776, 0.12121993561728593, 0.06391835304402342, 0.1874964844716252, 
0.31464720022011783)
(0.4570780157186622, 0.12442940577038865, 0.11029854307335335, 0.12366672320114874, 0.28901636685064414, 
0.3940638038809299)
3.4799485974083337'''
while changed:
    n_iteration += 1
    print('Cycle number:', n_iteration)
    calibration_model.run_calibration(initial_cases=n_cases, beta_range=[c_beta_inf, c_beta_base, c_beta_sup],
                                      death_range=[c_death_inf, c_death_base, c_death_sup],
                                      arrival_range=[c_arrival_inf, c_arrival_base, c_arrival_sup],
                                      dates={'days_cases': days_cases, 'days_deaths': days_deaths}, total=c_total,
                                      iteration=100+n_iteration, max_shrinks=5, max_no_improvement=50,
                                      min_value_to_iterate=1000)
    if calibration_model.current_results[0]['error'] < previous_error:
        changed = True
        previous_error = calibration_model.ideal_values['error']
        c_beta_base = calibration_model.ideal_values['beta']
        c_death_base = calibration_model.ideal_values['dc']
        c_arrival_base = calibration_model.ideal_values['arrival']
    else:
        changed = False

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
