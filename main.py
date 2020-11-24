from model_run import Model
from pyexcelerate import Workbook
from run_calibration_nelder_mead import Calibration
import pandas as pd
import datetime
import json
import time

model_ex = Model()
type_paramsA = dict()
for pv in model_ex.time_params:
    type_paramsA[pv] = 'BASE_VALUE'
for pv in model_ex.prob_params:
    type_paramsA[pv] = 'BASE_VALUE'

priority_vaccine_df = pd.read_csv('input\\priority_vaccines.csv', sep=';')
priority_vaccine_scenarios = dict()
for index, row in priority_vaccine_df.iterrows():
    priority_vaccine_scenarios[row['SCENARIO']] = priority_vaccine_scenarios.get(row['SCENARIO'], list())
    priority_vaccine_scenarios[row['SCENARIO']].append([row['AGE_GROUP'], row['WORK_GROUP'], row['HEALTH_GROUP']])
del priority_vaccine_df

vaccine_effectiveness_df = pd.read_csv('input\\vaccine_effectiveness.csv', sep=';')
vaccine_effectiveness_scenarios = dict()
for index, row in vaccine_effectiveness_df.iterrows():
    vaccine_effectiveness_scenarios[row['SCENARIO']] = vaccine_effectiveness_scenarios.get(row['SCENARIO'], dict())
    vaccine_effectiveness_scenarios[row['SCENARIO']][(row['AGE_GROUP'], row['HEALTH_GROUP'])] = \
        {'VACCINE_EFFECTIVENESS_1': row['VACCINE_EFFECTIVENESS_1'],
    'VACCINE_EFFECTIVENESS_2': row['VACCINE_EFFECTIVENESS_2']}
del vaccine_effectiveness_df

vaccine_information = pd.read_csv('input\\region_capacities.csv', sep=';', index_col=0).to_dict(orient='index')

start_processing_s_t = time.process_time()
start_time_t = datetime.datetime.now()

c_beta_inf = 0.0
c_beta_base = 0.01  # 0.008140019590906994
c_beta_sup = 0.5
c_death_inf = 1.4
c_death_base = 1.5  # 1.9830377761558986
c_death_sup = 2.6
calibration_model = Calibration()
c_beta_ant = 0.0
c_death_ant = 0.0
n_iteration = 0
# 'Beta': 0.008140019590906994, 'DC': 1.9830377761558986, 'Error': 0.7050015812639594
n_cases = 150
c_total = True
while c_beta_ant != c_beta_base or c_death_ant != c_death_base:
    if len(calibration_model.ideal_values) > 0:
        c_beta_sup = (c_beta_base+c_beta_sup)/2
        c_beta_inf = (c_beta_base+c_beta_inf)/2
        c_death_sup = (c_death_base + c_death_sup) / 2
        c_death_sup = (c_death_sup + c_death_sup) / 2
    c_beta_ant = c_beta_base
    c_death_ant = c_death_base
    n_iteration += 1
    print('Cycle number:', n_iteration)
    r = calibration_model.run_calibration(initial_cases=n_cases, beta_inf=c_beta_inf, beta_base=c_beta_base,
                                          beta_sup=c_beta_sup, death_inf=c_death_inf, death_base=c_death_base,
                                          death_sup=c_death_sup, total=c_total, iteration=n_iteration, max_shrinks=2)
    c_beta_base = calibration_model.ideal_values['Beta']
    c_death_base = calibration_model.ideal_values['DC']
    n_cases = round(n_cases*0.9)
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
results_pd_c = pd.from_dict(calibration_model.results)
c_values = [results_pd_c.columns] + list(results_pd_c.values)
c_wb = Workbook()
c_wb.new_sheet('All_values', data=c_values)
c_wb.save('output\\calibration_nm_results_' + ('total' if c_total else 'new') + '.xlsx')
print('Excel ', 'output\\calibration_consolidated_results_2_' + ('total' if c_total else 'new') + '.xlsx', 'exported')

for pvs in priority_vaccine_scenarios:
    for pes in vaccine_effectiveness_scenarios:
        print('Priority scenario: ', pvs, ' Effectiveness scenario: ', pes)
        c_name = 'vac_priority_' + pvs + '_effectiveness_'+pvs
        model_ex.run(type_params=type_paramsA, name=c_name, run_type='vaccination', beta=c_beta_base,
                     death_coefficient=c_death_base, sim_length=365*3, vaccine_priority=priority_vaccine_scenarios[pvs],
                     vaccine_effectiveness=vaccine_effectiveness_scenarios[pes],
                     vaccine_capacities=vaccine_information['VACCINE_CAPACITY'],
                     vaccine_start_day=vaccine_information['VACCINE_START_DAY'],
                     vaccine_end_day=vaccine_information['VACCINE_END_DAY'])

c_name = 'no_vac'
model_ex.run(type_params=type_paramsA, name=c_name, run_type='no_vaccination', beta=c_beta_base,
             death_coefficient=c_death_base, sim_length=365*3)

for pv in type_paramsA:
    type_paramsB = type_paramsA.copy()
    for val in ['INF_VALUE', 'MAX_VALUE']:
        type_paramsB[pv] = val
        for pvs in priority_vaccine_scenarios:
            for pes in vaccine_effectiveness_scenarios:
                print('Priority scenario: ', pvs, ' Effectiveness scenario: ', pes)
                c_name = 'vac_priority_' + pvs + '_effectiveness_' + pvs + '_sensitivity_' + pv + '_' + val
                model_ex.run(type_params=type_paramsA, name=c_name, run_type='vaccination', beta=c_beta_base,
                             death_coefficient=c_death_base, sim_length=365 * 3,
                             vaccine_priority=priority_vaccine_scenarios[pvs],
                             vaccine_effectiveness=vaccine_effectiveness_scenarios[pes],
                             vaccine_capacities=vaccine_information['VACCINE_CAPACITY'],
                             vaccine_start_day=vaccine_information['VACCINE_START_DAY'],
                             vaccine_end_day=vaccine_information['VACCINE_END_DAY'])

        c_name = 'vac_priority_sensitivity_' + pv + '_' + val
        model_ex.run(type_params=type_paramsA, name=c_name, run_type='no_vaccination', beta=c_beta_base,
                     death_coefficient=c_death_base, sim_length=365 * 3)

