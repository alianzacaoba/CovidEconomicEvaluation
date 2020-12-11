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
'''Current best results:
beta : (0.016365807406708593, 0.015195432338911278, 0.014413577299103739, 0.015032450873958988, 0.014314009181017572, 
0.013347855520539915)
dc : (1.2292383181904216, 0.6713077384948367, 1.11718006011982, 1.1587520607570703, 0.983949846176943, 
1.2289271808837245)
arrival : (28.008438791881524, 13.937260565381386, 28.019878666248914, 22.635142336479415, 27.653506361712104, 
28.004883621948935)
error_cases : (0.1710738861634178, 0.028753950929803408, 0.07452207536930135, 0.05309936977211302, 0.11063336954097483, 
0.0779034265566981)
error_deaths : (0.2352147405621718, 0.044984896535603794, 0.09019176524800784, 0.08694932754570311, 0.16925082236381206, 
0.15474358373683328)
error : 0.09336848687926873'''
c_beta_base = (0.015181261753330323, 0.014821128683311566, 0.0131325938037465, 0.014696523899962706,
               0.013138902986450665, 0.011345074849436072)
c_death_base = (1.9185998688640198, 0.6986535116775661, 1.1603601931643293, 1.3459781524694627, 1.203913031582318,
                1.1724583888461928)
c_arrival_base = (29.486905916730898, 9.668472205980201, 54.75314641851789, 14.313713580902798, 63.217938809786816,
                  118.4026978484823)
''' beta : (0.015181261753330323, 0.014821128683311566, 0.0131325938037465, 0.014696523899962706, 0.013138902986450665,
 0.011345074849436072)
  dc : (1.9185998688640198, 0.6986535116775661, 1.1603601931643293, 1.3459781524694627, 1.203913031582318,
   1.1724583888461928)
  arrival : (29.486905916730898, 9.668472205980201, 54.75314641851789, 14.313713580902798, 63.217938809786816,
   118.4026978484823)
  error_cases : (0.33331200898839547, 0.1352669778890263, 0.16199087707645252, 0.16314406721047983, 0.17294755944478774,
   0.02092245175807404)
  error_deaths : (0.33139388898960365, 0.1458089169985403, 0.12813867593200765, 0.19535080011637734, 
  0.16542937364106938, 0.015676027794398896)
  error : 0.16443659431411323
'''

c_name = 'no_vac'
model_ex.run(type_params=type_paramsA, name=c_name, run_type='no_vaccination', beta=c_beta_base,
             arrival_coefficient=c_arrival_base, death_coefficient=c_death_base,
             sim_length=365*3, export_type='xlsx')

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
                             vaccine_start_day=vaccine_start_days,
                             vaccine_end_day=vaccine_end_days, sim_length=365 * 3, export_type='xlsx')

        c_name = 'vac_priority_sensitivity_' + pv + '_' + val
        model_ex.run(type_params=type_paramsB, name=c_name, run_type='no_vaccination', beta=c_beta_base,
                     death_coefficient=c_death_base, arrival_coefficient=c_arrival_base, sim_length=365 * 3,
                     export_type='xlsx')
