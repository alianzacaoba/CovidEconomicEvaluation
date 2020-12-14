import multiprocessing

from tqdm import tqdm

from logic.model import Model
import pandas as pd
from root import DIR_INPUT


def chunks(lv, nv):
    for iv in range(0, len(lv), nv):
        yield lv[iv:iv + nv]


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

c_beta_base = (0.015181261753330323, 0.014821128683311566, 0.0131325938037465, 0.014696523899962706,
               0.013138902986450665, 0.011345074849436072)
c_death_base = (1.9185998688640198, 0.6986535116775661, 1.1603601931643293, 1.3459781524694627, 1.203913031582318,
                1.1724583888461928)
c_arrival_base = (29.486905916730898, 9.668472205980201, 54.75314641851789, 14.313713580902798, 63.217938809786816,
                  118.4026978484823)

manager = multiprocessing.Manager()
return_list = manager.list()
jobs = list()
cores = multiprocessing.cpu_count() - 1
p = multiprocessing.Process(target=model_ex.run, args=(type_paramsA, 'no_vac', 'no_vaccination', c_beta_base,
                                                       c_death_base, c_arrival_base, None, None, None, None, None,
                                                       (365 * 3), 'all', 1.0, False))
jobs.append(p)
for pvs in priority_vaccine_scenarios:
    for pes in vaccine_effectiveness_scenarios:
        print('Priority scenario: ', pvs, ' Effectiveness scenario: ', pes)
        c_name = 'vac_priority_' + str(pvs) + '_effectiveness_' + str(pes)
        p = multiprocessing.Process(target=model_ex.run, args=(type_paramsA, c_name, 'no_vaccination', c_beta_base,
                                                               c_death_base, c_arrival_base,
                                                               priority_vaccine_scenarios[pvs],
                                                               vaccine_information['VACCINE_CAPACITY'],
                                                               vaccine_effectiveness_scenarios[pes], vaccine_start_days,
                                                               vaccine_end_days, (365 * 3), 'all', 1.0, False))
        jobs.append(p)

for pv in type_paramsA:
    type_paramsB = type_paramsA.copy()
    for val in ['INF_VALUE', 'MAX_VALUE']:
        type_paramsB[pv] = val
        c_name = 'sensitivity_' + pv + '_' + val + '_vac_priority_no_vac'
        p = multiprocessing.Process(target=model_ex.run, args=(type_paramsB, 'no_vac', 'no_vaccination', c_beta_base,
                                                               c_death_base, c_arrival_base, None, None, None, None,
                                                               None,
                                                               (365 * 3), 'all', 1.0, False))
        jobs.append(p)
        for pvs in priority_vaccine_scenarios:
            for pes in vaccine_effectiveness_scenarios:
                print('Priority scenario: ', pvs, ' Effectiveness scenario: ', pes)
                c_name = 'sensitivity_' + pv + '_' + val + '_vac_priority_' + str(pvs) + '_effectiveness_' + str(pes)
                p = multiprocessing.Process(target=model_ex.run,
                                            args=(type_paramsB, c_name, 'no_vaccination', c_beta_base,
                                                  c_death_base, c_arrival_base,
                                                  priority_vaccine_scenarios[pvs],
                                                  vaccine_information['VACCINE_CAPACITY'],
                                                  vaccine_effectiveness_scenarios[pes], vaccine_start_days,
                                                  vaccine_end_days, (365 * 3), 'all', 1.0, False))
                jobs.append(p)

        c_name = 'vac_priority_sensitivity_' + pv + '_' + val
        model_ex.run(type_params=type_paramsB, name=c_name, run_type='no_vaccination', beta=c_beta_base,
                     death_coefficient=c_death_base, arrival_coefficient=c_arrival_base, sim_length=365 * 3,
                     export_type='xlsx')

for i in tqdm(chunks(jobs, cores)):
    for j in i:
        j.start()
    for j in i:
        j.join()
