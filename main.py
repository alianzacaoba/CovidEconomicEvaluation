from model import Model
import pandas as pd


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

c_daly_vector = {'Home': 0.2, 'Hospital': 0.3, 'ICU': 0.5, 'Death': 1, 'Recovered': 0.1}

vaccine_information = pd.read_csv('input\\region_capacities.csv', sep=';', index_col=0).to_dict()
# {'Beta': 0.00742441464418964, 'DC': 2.128698267410927, 'Error': 0.5942430917833809}
c_beta_base = 0.00742441464418964
c_death_base = 2.128698267410927

for pvs in priority_vaccine_scenarios:
    for pes in vaccine_effectiveness_scenarios:
        print('Priority scenario: ', pvs, ' Effectiveness scenario: ', pes)
        c_name = 'vac_priority_' + str(pvs) + '_effectiveness_' + str(pes)
        model_ex.run(type_params=type_paramsA, name=c_name, run_type='vaccination', beta=c_beta_base,
                     death_coefficient=c_death_base, sim_length=365*3, vaccine_priority=priority_vaccine_scenarios[pvs],
                     daly_vector = c_daly_vector, vaccine_effectiveness=vaccine_effectiveness_scenarios[pes],
                     vaccine_capacities=vaccine_information['VACCINE_CAPACITY'],
                     vaccine_start_day=vaccine_information['VACCINE_START_DAY'],
                     vaccine_end_day=vaccine_information['VACCINE_END_DAY'])

c_name = 'no_vac'
model_ex.run(type_params=type_paramsA, name=c_name, run_type='no_vaccination', beta=c_beta_base,
             daly_vector = c_daly_vector, death_coefficient=c_death_base, sim_length=365*3)

for pv in type_paramsA:
    type_paramsB = type_paramsA.copy()
    for val in ['INF_VALUE', 'MAX_VALUE']:
        type_paramsB[pv] = val
        for pvs in priority_vaccine_scenarios:
            for pes in vaccine_effectiveness_scenarios:
                print('Priority scenario: ', pvs, ' Effectiveness scenario: ', pes)
                c_name = 'vac_priority_' + str(pvs) + '_effectiveness_' + str(pes) + '_sensitivity_' + pv + '_' + val
                model_ex.run(type_params=type_paramsA, name=c_name, run_type='vaccination', beta=c_beta_base,
                             death_coefficient=c_death_base, sim_length=365 * 3, daly_vector = c_daly_vector,
                             vaccine_priority=priority_vaccine_scenarios[pvs],
                             vaccine_effectiveness=vaccine_effectiveness_scenarios[pes],
                             vaccine_capacities=vaccine_information['VACCINE_CAPACITY'],
                             vaccine_start_day=vaccine_information['VACCINE_START_DAY'],
                             vaccine_end_day=vaccine_information['VACCINE_END_DAY'])

        c_name = 'vac_priority_sensitivity_' + pv + '_' + val
        model_ex.run(type_params=type_paramsA, name=c_name, run_type='no_vaccination', beta=c_beta_base,
                     daly_vector = c_daly_vector, death_coefficient=c_death_base, sim_length=365 * 3)
