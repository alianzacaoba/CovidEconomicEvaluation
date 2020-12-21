import multiprocessing
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
beta : (0.021491363723845736, 0.0166199610610343, 0.01791884096730873, 0.017498833026023182, 0.017314005344102094, 0.016117804628266723)
dc : (1.0512197892987984, 0.8584458269048881, 1.2588115965945355, 0.949138246970027, 0.8338365674750021, 0.5123820859223633)
arrival : (13.746462972480563, 13.735298422082789, 11.600253075852974, 11.406959225912638, 16.459230005367647, 16.369257067719225)
spc : 0.30746153001928733
error_cases : (0.017132066221389804, 0.0698091087662983, 0.030043995987563386, 0.01818317845395831, 0.05394426674254785, 0.05763890749701983)
error_deaths : (0.03479641142249817, 0.022013661349029844, 0.05378338430507586, 0.03156197354146263, 0.06382440220255423, 0.24813059659514178)
error : 0.04688522354887917
'''

c_beta_base = (0.021491363723845736, 0.0166199610610343, 0.01791884096730873, 0.017498833026023182,
               0.017314005344102094, 0.016117804628266723)
c_death_base = (1.0512197892987984, 0.8584458269048881, 1.2588115965945355, 0.949138246970027, 0.8338365674750021,
                0.5123820859223633)
c_arrival_base = (13.746462972480563, 13.735298422082789, 11.600253075852974, 11.406959225912638, 16.459230005367647,
                  16.369257067719225)
spc = 0.30746153001928733

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
