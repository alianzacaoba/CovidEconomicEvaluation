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

''' beta : (0.01809861666454559, 0.016112349678480548, 0.015592823242052592, 0.016334099695781, 0.015594377165214989, 
0.014149377755879794)
  dc : (1.196779691452801, 0.8578036308722543, 1.1613525392820707, 1.1720402602369706, 1.0085875953511478, 
  1.123814372496263)
  arrival : (12.858228810965814, 5.140364799835279, 15.70021265510659, 7.366994654680764, 18.716507863968452, 
  18.74379230948766)
  error_cases : (0.22115220410819905, 0.059111098849936976, 0.1184983895238421, 0.06269524698350858, 
  0.14637034210632907, 0.2759108454680301)
  error_deaths : (0.31659541626907284, 0.10385117267943263, 0.09262767343074498, 0.09156097588124046, 
  0.24359750256480622, 0.3558797217427209)
  error : 2.9715887166477097
'''
c_beta_base = [0.01809861666454559, 0.016112349678480548, 0.015592823242052592, 0.016334099695781, 0.015594377165214989,
               0.014149377755879794]
c_death_base = [1.196779691452801, 0.8578036308722543, 1.1613525392820707, 1.1720402602369706, 1.0085875953511478,
                1.123814372496263]
c_arrival_base = [12.858228810965814, 5.140364799835279, 15.70021265510659, 7.366994654680764, 18.716507863968452,
                  18.74379230948766]

c_name = 'no_vac'
model_ex.run(type_params=type_paramsA, name=c_name, run_type='no_vaccination', beta=c_beta_base,
             arrival_coefficient=c_arrival_base, death_coefficient=c_death_base,
             sim_length=365*3)

for pvs in priority_vaccine_scenarios:
    for pes in vaccine_effectiveness_scenarios:
        print('Priority scenario: ', pvs, ' Effectiveness scenario: ', pes)
        c_name = 'vac_priority_' + str(pvs) + '_effectiveness_' + str(pes)
        model_ex.run(type_params=type_paramsA, name=c_name, run_type='vaccination', beta=c_beta_base,
                     death_coefficient=c_death_base, arrival_coefficient=c_arrival_base,
                     vaccine_priority=priority_vaccine_scenarios[pvs],
                     vaccine_capacities=vaccine_information['VACCINE_CAPACITY'],
                     vaccine_effectiveness=vaccine_effectiveness_scenarios[pes],
                     vaccine_start_day=vaccine_start_days,
                     vaccine_end_day=vaccine_end_days, sim_length=365 * 3)

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
                             vaccine_start_day=vaccine_start_days,
                             vaccine_end_day=vaccine_end_days, sim_length=365 * 3)

        c_name = 'vac_priority_sensitivity_' + pv + '_' + val
        model_ex.run(type_params=type_paramsB, name=c_name, run_type='no_vaccination', beta=c_beta_base,
                     death_coefficient=c_death_base, arrival_coefficient=c_arrival_base, sim_length=365 * 3)
