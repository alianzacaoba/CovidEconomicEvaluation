import multiprocessing
from tqdm import tqdm
from logic.model import Model
import pandas as pd
from root import DIR_INPUT
import time


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

''' Current best results:
beta : (0.022007606119173795, 0.017863980243733434, 0.018196130566806898, 0.018457626077325776, 0.01748616435199459, 
0.016227115077950355)
dc : (1.1323965306925503, 0.7125429486836731, 1.0444436705577909, 1.0455277531926157, 0.8937395613665182, 
0.9248158502415792)
arrival : (15.610984192361858, 7.118033263153407, 13.580052334837838, 6.872622856121195, 19.179202373513895, 
23.821317070305813)
spc : 0.2761426379037166
error_cases : (0.01144914782225527, 0.04633153074628225, 0.035662564790543826, 0.01663383341316379, 
0.044033713992227476, 0.05507748681553992)
error_deaths : (0.027622229848766704, 0.009587082330408638, 0.012646483805184615, 0.01921871618354314, 
0.062333297880331026, 0.07248222923337995)
error : 0.03471753964393546
'''

c_beta_base = (0.022007606119173795, 0.017863980243733434, 0.018196130566806898, 0.018457626077325776,
               0.01748616435199459, 0.016227115077950355)
c_death_base = (1.1323965306925503, 0.7125429486836731, 1.0444436705577909, 1.0455277531926157, 0.8937395613665182,
                0.9248158502415792)
c_arrival_base = (15.610984192361858, 7.118033263153407, 13.580052334837838, 6.872622856121195, 19.179202373513895,
                  23.821317070305813)
spc = 0.2761426379037166

manager = multiprocessing.Manager()
return_list = manager.list()
jobs = list()
cores = multiprocessing.cpu_count() - 1
p = multiprocessing.Process(target=model_ex.run, args=(type_paramsA, 'no_vac', 'no_vaccination', c_beta_base,
                                                       c_death_base, c_arrival_base, spc, None, None, None, None, None,
                                                       (365 * 3), 'csv', False))
jobs.append(p)
jobs[0].start()
for pvs in priority_vaccine_scenarios:
    for pes in vaccine_effectiveness_scenarios:
        c_name = 'vac_priority_' + str(pvs) + '_effectiveness_' + str(pes)
        p = multiprocessing.Process(target=model_ex.run, args=(type_paramsA, c_name, 'no_vaccination', c_beta_base,
                                                               c_death_base, c_arrival_base, spc,
                                                               priority_vaccine_scenarios[pvs],
                                                               vaccine_information['VACCINE_CAPACITY'],
                                                               vaccine_effectiveness_scenarios[pes], vaccine_start_days,
                                                               vaccine_end_days, (365 * 3), 'csv', False))
        available = False
        while not available:
            n_count = 0
            for j in range(len(jobs)):
                if jobs[j].is_alive():
                    n_count += 1
            if n_count < cores:
                print(c_name)
                available = True
                jobs.append(p)
                jobs[len(jobs) - 1].start()
            else:
                time.sleep(1)

for pv in type_paramsA:
    type_paramsB = type_paramsA.copy()
    for val in ['INF_VALUE', 'MAX_VALUE']:
        type_paramsB[pv] = val
        c_name = 'sensitivity_' + pv + '_' + val + '_vac_priority_no_vac'
        p = multiprocessing.Process(target=model_ex.run, args=(type_paramsB, c_name, 'no_vaccination', c_beta_base,
                                                               c_death_base, c_arrival_base, spc, None, None, None,
                                                               None, None, (365 * 3), 'csv', False))
        available = False
        while not available:
            n_count = 0
            for j in range(len(jobs)):
                if jobs[j].is_alive():
                    n_count += 1
            if n_count < cores:
                print(c_name)
                available = True
                jobs.append(p)
                jobs[len(jobs) - 1].start()
            else:
                time.sleep(1)

        for pvs in priority_vaccine_scenarios:
            for pes in vaccine_effectiveness_scenarios:
                c_name = 'sensitivity_' + pv + '_' + val + '_vac_priority_' + str(pvs) + '_effectiveness_' + str(pes)
                p = multiprocessing.Process(target=model_ex.run,
                                            args=(type_paramsB, c_name, 'no_vaccination', c_beta_base, c_death_base,
                                                  c_arrival_base, spc, priority_vaccine_scenarios[pvs],
                                                  vaccine_information['VACCINE_CAPACITY'],
                                                  vaccine_effectiveness_scenarios[pes], vaccine_start_days,
                                                  vaccine_end_days, (365 * 3), 'csv', False))
                available = False
                while not available:
                    n_count = 0
                    for j in range(len(jobs)):
                        if jobs[j].is_alive():
                            n_count += 1
                    if n_count < cores:
                        print(c_name)
                        available = True
                        jobs.append(p)
                        jobs[len(jobs) - 1].start()
                    else:
                        time.sleep(1)

available = False
while not available:
    n_count = 0
    for j in range(len(jobs)):
        if jobs[j].is_alive():
            n_count += 1
    if n_count == 0:
        available = True
    else:
        time.sleep(1)
print('End process')
