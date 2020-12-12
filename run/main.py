from logic.model import Model
import pandas as pd

from root import DIR_INPUT

model_ex = Model()
type_paramsA = dict()
for pv in model_ex.time_params:
    type_paramsA[pv[0]] = 'BASE_VALUE'
for pv in model_ex.prob_params:
    type_paramsA[pv[0]] = 'BASE_VALUE'

priority_vaccine_df = pd.read_csv(DIR_INPUT + 'priority_vaccines.csv', sep=';')
priority_vaccine_scenarios = dict()
for index, row in priority_vaccine_df.iterrows():
    priority_vaccine_scenarios[row['SCENARIO']] = priority_vaccine_scenarios.get(row['SCENARIO'], list())
    priority_vaccine_scenarios[row['SCENARIO']].append([row['AGE_GROUP'], row['WORK_GROUP'], row['HEALTH_GROUP']])
del priority_vaccine_df

vaccine_effectiveness_df = pd.read_csv(DIR_INPUT + 'vaccine_effectiveness.csv', sep=';')
vaccine_effectiveness_scenarios = dict()
for index, row in vaccine_effectiveness_df.iterrows():
    vaccine_effectiveness_scenarios[row['SCENARIO']] = vaccine_effectiveness_scenarios.get(row['SCENARIO'], dict())
    vaccine_effectiveness_scenarios[row['SCENARIO']][(row['AGE_GROUP'], row['HEALTH_GROUP'])] = \
        {'VACCINE_EFFECTIVENESS_1': row['VACCINE_EFFECTIVENESS_1'],
    'VACCINE_EFFECTIVENESS_2': row['VACCINE_EFFECTIVENESS_2']}
del vaccine_effectiveness_df

vaccine_start_days = {'INF_VALUE': 377, 'BASE_VALUE': 419, 'MAX_VALUE': 480}
vaccine_end_days = {'INF_VALUE': 682, 'BASE_VALUE': 724, 'MAX_VALUE': 785}
n_vaccine_days = 305
type_paramsA['daly'] = 'BASE_VALUE'
type_paramsA['cost'] = 'BASE_VALUE'
type_paramsA['vaccine_day'] = 'BASE_VALUE'

vaccine_information = pd.read_csv(DIR_INPUT + 'region_capacities.csv', sep=';', index_col=0).to_dict()
c_beta_base = (0.023572152155186078, 0.01769950655220088, 0.018519129301769154, 0.016885653990396418,
               0.019242207702252524, 0.018371376307632463)
c_death_base = (1.1963156938281962, 0.6502100996039765, 1.081336086898889, 1.2182665596973046, 0.808702873981918,
                0.8452165571388334)
c_arrival_base = (16.0697388326682, 14.816066973348956, 8.958040652870444, 28.976417785691453, 12.803199978893517,
                  15.008478319271791)
''' beta : (0.023572152155186078, 0.01769950655220088, 0.018519129301769154, 0.016885653990396418, 0.019242207702252524, 
0.018371376307632463)
dc : (1.1963156938281962, 0.6502100996039765, 1.081336086898889, 1.2182665596973046, 0.808702873981918, 
0.8452165571388334)
arrival : (16.0697388326682, 14.816066973348956, 8.958040652870444, 28.976417785691453, 12.803199978893517, 
15.008478319271791)
spc : 0.22806022589268016
error_cases : (0.030731576927313008, 0.13487872924890876, 0.08983902208839513, 0.09503011221461172, 0.04795620001158757, 
0.0948414431119049)
error_deaths : (0.008303319541706095, 0.05445524250758637, 0.045647159114592474, 0.026211759463199154, 
0.03972596211634231, 0.05041162499841424)
error : 0.07475390238209573
'''

c_name = 'no_vac'
model_ex.run(type_params=type_paramsA, name=c_name, run_type='no_vaccination', beta=c_beta_base,
             death_coefficient=c_death_base, arrival_coefficient=c_arrival_base, sim_length=365 * 3, export_type='xlsx')

for pvs in priority_vaccine_scenarios:
    for pes in vaccine_effectiveness_scenarios:
        print('Priority scenario: ', pvs, ' Effectiveness scenario: ', pes)
        c_name = 'vac_priority_' + str(pvs) + '_effectiveness_' + str(pes)
        model_ex.run(type_params=type_paramsA, name=c_name, run_type='vaccination', beta=c_beta_base,
                     death_coefficient=c_death_base, arrival_coefficient=c_arrival_base,
                     vaccine_priority=priority_vaccine_scenarios[pvs],
                     vaccine_capacities=vaccine_information['VACCINE_CAPACITY'],
                     vaccine_effectiveness=vaccine_effectiveness_scenarios[pes], vaccine_start_day=vaccine_start_days,
                     vaccine_end_day=vaccine_end_days, sim_length=365 * 3, export_type='xlsx')

for pv in type_paramsA:
    type_paramsB = type_paramsA.copy()
    for val in ['INF_VALUE', 'MAX_VALUE']:
        type_paramsB[pv] = val
        for pvs in priority_vaccine_scenarios:
            for pes in vaccine_effectiveness_scenarios:
                print('Priority scenario: ', pvs, ' Effectiveness scenario: ', pes)
                c_name = 'vac_priority_' + str(pvs) + '_effectiveness_' + str(pes) + '_sensitivity_' + pv + '_' + val
                model_ex.run(type_params=type_paramsB, name=c_name, run_type='vaccination', beta=c_beta_base,
                             death_coefficient=c_death_base, arrival_coefficient=c_arrival_base,
                             vaccine_priority=priority_vaccine_scenarios[pvs],
                             vaccine_capacities=vaccine_information['VACCINE_CAPACITY'],
                             vaccine_effectiveness=vaccine_effectiveness_scenarios[pes],
                             vaccine_start_day=vaccine_start_days, vaccine_end_day=vaccine_end_days, sim_length=365 * 3,
                             export_type='xlsx')

        c_name = 'vac_priority_sensitivity_' + pv + '_' + val
        model_ex.run(type_params=type_paramsB, name=c_name, run_type='no_vaccination', beta=c_beta_base,
                     death_coefficient=c_death_base, arrival_coefficient=c_arrival_base, sim_length=365 * 3,
                     export_type='xlsx')
