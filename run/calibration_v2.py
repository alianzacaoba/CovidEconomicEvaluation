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

'''   beta : (0.01519106865233014, 0.014834054096359864, 0.01314227928274465, 0.014725585185679964, 
0.013181919300454894, 0.01134869046564806)
dc : (1.9495951213564333, 0.7030284120006993, 1.1678491345140833, 1.3369167259796195, 1.2015019218211047, 
1.1745292453624234)
arrival : (28.668970477478446, 9.453307190442839, 53.26641924059571, 13.882825378228919, 59.6260936536479, 
117.90863544243571)
error_cases : (0.3329469711040809, 0.1350899341678179, 0.16171605900424274, 0.16285190196632965, 0.17231470188386155, 
0.0209214539493001)
error_deaths : (0.334718424981469, 0.14740248871093176, 0.1305547142629635, 0.19787549556127337, 0.1733229750509208, 
0.015894479966259617)
error : 0.16469371358088838
'''

c_beta_base = np.array((0.015182300803015896, 0.014832950234415509, 0.013375639032321248, 0.014727687703501257,
                        0.013446269427707706, 0.012327379379101604))
c_beta_inf = c_beta_base*0.9
c_beta_sup = c_beta_base*1.1
c_death_base = np.array((1.9591990540739632, 0.730123854374982, 1.1324641871372727, 1.3363779777352005,
                         1.4117011249830445, 1.4655163432891358))
c_death_inf = c_death_base*.9
c_death_sup = c_death_base*1.1
c_arrival_inf = np.ones(6)*1.0
c_arrival_base = np.array((28.83618741570408, 9.22862115530603, 39.90057003805639, 13.878468430957831,
                           40.778047581453556, 39.15099032135454))
c_arrival_sup = np.ones(6)*45

calibration_model = Calibration()
c_beta_ant = 0.0
c_death_ant = 0.0
n_iteration = 0
days_deaths = {1: 62, 2: 105, 3: 47, 4: 111, 5: 58, 6: 84}
days_cases = {1: 23, 2: 20, 3: 18, 4: 30, 5: 22, 6: 55}
n_cases = 5*(len(c_beta_base) + len(c_death_base) + len(c_arrival_base))+1
c_total = True
n_changed = 0
previous_error = [100000000000000.0]
while n_changed < 2:
    n_iteration += 1
    print('Cycle number:', n_iteration)
    calibration_model.run_calibration(initial_cases=n_cases, beta_range=[c_beta_inf, c_beta_base, c_beta_sup],
                                      death_range=[c_death_inf, c_death_base, c_death_sup],
                                      arrival_range=[c_arrival_inf, c_arrival_base, c_arrival_sup],
                                      dates={'days_cases': days_cases, 'days_deaths': days_deaths}, total=c_total,
                                      iteration=10000+n_iteration, max_no_improvement=50,
                                      min_value_to_iterate=1000, weights=[1, 1, 1])
    print(' End of cycle: ', n_iteration)
    print(' New error:', calibration_model.ideal_values['error'])
    print(' Previous errors:', previous_error)
    print(' Improvement: ', float(previous_error[len(previous_error) - 1] -
                                  calibration_model.ideal_values['error']))
    print(' No changes in: ', n_changed)
    if float(calibration_model.ideal_values['error']) < float(previous_error[len(previous_error) - 1]):
        n_changed = 0
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
with open(DIR_OUTPUT + 'calibration_consolidated_results_6_' + ('total' if c_total else 'new') + '.json', 'w') as fp:
    json.dump(calibration_model.results, fp)
results_pd_c = pd.DataFrame(calibration_model.results)
c_values = [results_pd_c.columns] + list(results_pd_c.values)
c_wb = Workbook()
c_wb.new_sheet('All_values', data=c_values)
c_wb.save(DIR_OUTPUT + 'calibration_consolidated_results_6_' + ('total' if c_total else 'new') + '.xlsx')
print('Excel ', DIR_OUTPUT + 'calibration_consolidated_results_6_' + ('total' if c_total else 'new') + '.xlsx',
      'exported')
