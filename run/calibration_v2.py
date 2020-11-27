from pyexcelerate import Workbook
from logic.calibration_nelder_mead import Calibration
import pandas as pd
import numpy as np
import datetime
import json
import time

start_processing_s_t = time.process_time()
start_time_t = datetime.datetime.now()

c_beta_inf = np.zeros(6)
c_beta_base = np.ones(6)*0.1  # 0.008140019590906994
c_beta_sup = np.ones(6)*0.9
c_death_inf = np.ones(6)
c_death_base = np.ones(6)*1.5  # 1.9830377761558986
c_death_sup = np.ones(6)*2.2
c_arrival_inf = np.ones(6)*1.0
c_arrival_base = np.ones(6)*1.0
c_arrival_sup = np.ones(6)*10
calibration_model = Calibration()
c_beta_ant = 0.0
c_death_ant = 0.0
n_iteration = 0
days_deaths = {1: 62, 2: 105, 3: 47, 4: 111, 5: 58, 6: 84}
days_cases = {1: 23, 2: 20, 3: 18, 4: 30, 5: 22, 6: 55}
# {'beta': [0.010186860018368264, 0.01025999182286686, 0.01012861550166681, 0.010061765037031652, 0.010221876308953864,
# 0.010134345468388704], 'dc': [1.5548150639242488, 1.4946730745696455, 1.4799855418256394, 1.463417261594141,
# 1.4848669979831661, 1.5123208592164006], 'arrival': [4.91557222639614, 5.079920417373481, 5.127148868703013,
# 5.214650388469965, 5.074283649728818, 4.9715447323387725], 'error_cases': 4.883773174533946, 'error_deaths':
# 5.07497126257022, 'error': 9.958744437104166}
n_cases = 10*(len(c_beta_base) + len(c_death_base) + len(c_arrival_base))+1
c_total = True
changed = True
error_ant = 100000000000000
while changed:
    n_iteration += 1
    print('Cycle number:', n_iteration)
    calibration_model.run_calibration(initial_cases=n_cases, beta_range=[c_beta_inf, c_beta_base, c_beta_sup],
                                      death_range=[c_death_inf, c_death_base, c_death_sup],
                                      arrival_range=[c_arrival_inf, c_arrival_base, c_arrival_sup],
                                      dates={'days_cases': days_cases, 'days_deaths': days_deaths}, total=c_total,
                                      iteration=n_iteration+100, max_shrinks=5, max_no_improvement=50,
                                      min_value_to_iterate=1000)
    c_arrival_base += 1
    changed = (error_ant > calibration_model.ideal_values['error'])
    error_ant = calibration_model.ideal_values['error']
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
with open('output\\calibration_consolidated_results_2' + ('total' if c_total else 'new') + '.json', 'w') as fp:
    json.dump(calibration_model.results, fp)
results_pd_c = pd.DataFrame(calibration_model.results)
c_values = [results_pd_c.columns] + list(results_pd_c.values)
c_wb = Workbook()
c_wb.new_sheet('All_values', data=c_values)
c_wb.save('output\\calibration_nm_results_' + ('total' if c_total else 'new') + '.xlsx')
print('Excel ', 'output\\calibration_consolidated_results_2_' + ('total' if c_total else 'new') + '.xlsx', 'exported')
