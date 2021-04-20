import multiprocessing

from tqdm import tqdm

from logic.model import Model
from pyexcelerate import Workbook
import numpy as np
import pandas as pd
import datetime
import json
import time
from root import DIR_INPUT, DIR_OUTPUT


def chunks(lv, nv):
    sets = []
    n_list = []
    for i in lv:
        n_list.append(i)
        if len(n_list) == nv:
            sets.append(n_list.copy())
            n_list = []
    if len(n_list) > 0:
        sets.append(n_list)
    return sets


class Calibration(object):

    def __init__(self):
        self.model = Model()
        self.ideal_values = None
        self.results = list()
        self.current_results = list()
        self.type_params = dict()
        for pv in self.model.time_params:
            self.type_params[pv[0]] = 'BASE_VALUE'
        for pv in self.model.prob_params:
            self.type_params[pv[0]] = 'BASE_VALUE'
        self.type_params['daly'] = 'BASE_VALUE'
        self.type_params['cost'] = 'BASE_VALUE'
        self.real_cases = pd.read_csv(DIR_INPUT + 'real_cases.csv')
        self.real_deaths = pd.read_csv(DIR_INPUT + 'death_cases.csv')
        self.real_seroprevalence = pd.read_csv(DIR_INPUT + 'seroprevalence.csv', index_col=0).to_dict('index')
        self.max_day = int(self.real_cases.SYM_DAY.max())
        self.error_ratio = 10
        self.error_power = 2

    def calculate_point(self, real_seroprevalence: dict, real_case: np.array, real_death: np.array, weights: tuple,
                        beta: tuple, dc: tuple, arrival: tuple, symptomatic_probability: float, dates: dict,
                        end_list: list, total: bool = True):

        sim_results = self.model.run(self.type_params, name='Calibration1', run_type='calibration', beta=beta,
                                     death_coefficient=dc, arrival_coefficient=arrival, sim_length=self.max_day,
                                     symptomatic_coefficient=symptomatic_probability, use_tqdm=False)
        error_seroprevalence = list()
        for reg in range(1, 7):
            real_point = real_seroprevalence[reg]['SEROPREVALENCE']
            error_seroprevalence.append(abs(real_point - sim_results[real_seroprevalence[reg]['SYM_DAY'], 11 + reg])
                                        / real_point)

        if not total:
            sim_results_alt = sim_results.copy()
            for k in range(1, len(sim_results)):
                sim_results[k] = sim_results_alt[k] - sim_results_alt[k - 1]
            del sim_results_alt
        error_cases = list()
        for reg in range(6):
            weight = np.array([(i+1-dates['days_cases'][reg + 1])**self.error_power
                               for i in range(dates['days_cases'][reg + 1], self.max_day+1)]) /\
                     sum((i+1-dates['days_cases'][reg + 1])**self.error_power
                         for i in range(dates['days_cases'][reg + 1], self.max_day+1))
            current_error = np.power(sim_results[dates['days_cases'][reg + 1]:, reg] /
                                     real_case[dates['days_cases'][reg + 1]:, reg] - 1, 2)
            error_cases.append(np.sum(np.multiply(weight, current_error)))
        error_deaths = list()
        for reg in range(6):
            weight = np.array([(i + 1 - dates['days_deaths'][reg + 1])**self.error_power
                               for i in range(dates['days_deaths'][reg + 1], self.max_day+1)]) / \
                     sum((i + 1 - dates['days_deaths'][reg + 1])**self.error_power for i in
                         range(dates['days_deaths'][reg + 1], self.max_day+1))
            current_error = np.power(sim_results[dates['days_deaths'][reg + 1]:, 6 + reg] /
                                     real_death[dates['days_deaths'][reg + 1]:, reg] - 1, 2)
            error_deaths.append(np.sum(np.multiply(weight, current_error)))
        error = (weights[0]*sum(error_seroprevalence) + weights[1] * sum(error_cases) + weights[2]*sum(error_deaths)) /\
                (6*sum(weights))
        end_list.append({'beta': tuple(beta), 'dc': tuple(dc), 'arrival': tuple(arrival),
                         'spc': symptomatic_probability, 'error_seroprevalence': tuple(error_seroprevalence),
                         'error_cases': tuple(error_cases), 'error_deaths': tuple(error_deaths), 'error': error})

    def try_new_best(self, new_point: dict, real_seroprevalence: dict, real_case: np.array, real_death: np.array,
                     weights: tuple, dates: dict, result_list: list, total: bool = True):
        beta = list()
        dc = list()
        arrival = list()
        changed = False
        number = max(multiprocessing.cpu_count()-1, 2)
        for reg in range(6):
            if new_point['error_cases'][reg] < self.ideal_values['error_cases'][reg] or \
                    new_point['error_seroprevalence'][reg] < self.ideal_values['error_seroprevalence'][reg]:
                beta.append(new_point['beta'][reg])
                arrival.append(new_point['arrival'][reg])
                if new_point['error_deaths'][reg] < self.ideal_values['error_deaths'][reg]:
                    dc.append(new_point['dc'][reg])
                else:
                    try:
                        dc.append(self.ideal_values['dc'][reg]*new_point['beta'][reg]/self.ideal_values['beta'][reg])
                    except Exception as exc:
                        dc.append(new_point['dc'][reg])
                        print(exc)
                changed = True
            else:
                beta.append(self.ideal_values['beta'][reg])
                arrival.append(self.ideal_values['arrival'][reg])
                if new_point['error_deaths'][reg] < self.ideal_values['error_deaths'][reg] and new_point['beta'][reg] \
                        > 0.0:
                    try:
                        dc.append(new_point['dc'][reg]*self.ideal_values['beta'][reg]/new_point['beta'][reg])
                    except:
                        dc.append(new_point['dc'][reg])
                    changed = True
                else:
                    dc.append(self.ideal_values['dc'][reg])
        if changed:
            beta = tuple(beta)
            dc = tuple(dc)
            arrival = tuple(arrival)
            points = list()
            min_val = min(self.ideal_values['spc'], new_point['spc'])
            max_val = max(self.ideal_values['spc'], new_point['spc'])
            dif = (max_val-min_val)/(number-1)
            for val in range(number):
                exists = False
                current_spc = min_val+dif*val
                for point in self.current_results:
                    if point['beta'] == beta and point['dc'] == dc and point['arrival'] == arrival and point['spc'] ==\
                            current_spc:
                        exists = True
                if not exists:
                    points.append((real_seroprevalence, real_case, real_death, weights, beta, dc, arrival, current_spc,
                                   dates, result_list, total))
            return points
        return []

    def run_calibration(self, beta_range: list, death_range: list, arrival_range: list,
                        symptomatic_probability_range: list, dates: dict, dimensions: int = 19, initial_cases: int = 30,
                        total: bool = True, iteration: int = 1, max_no_improvement: int = 100,
                        min_value_to_iterate: float = 10000.0, error_precision: int = 7, weights: list = (1, 1, 1)):
        # Weights: [Seroprevalence, Cases, Deaths]
        start_processing_s = time.process_time()
        start_time = datetime.datetime.now()
        real_seroprevalence = self.real_seroprevalence
        real_case = np.array((self.real_cases[['TOTAL_1', 'TOTAL_2', 'TOTAL_3', 'TOTAL_4', 'TOTAL_5', 'TOTAL_6']]
                              if total else self.real_cases[['NEW_1', 'NEW_2', 'NEW_3', 'NEW_4', 'NEW_5', 'NEW_6']]),
                             dtype='float64')
        real_death = np.array((self.real_deaths[['TOTAL_1', 'TOTAL_2', 'TOTAL_3', 'TOTAL_4', 'TOTAL_5', 'TOTAL_6']]
                               if total else self.real_deaths[['NEW_1', 'NEW_2', 'NEW_3', 'NEW_4', 'NEW_5', 'NEW_6']]),
                              dtype='float64')
        x_beta = list()
        x_d_c = list()
        x_arrival = list()
        beta = list()
        dc = list()
        arrival = list()
        symptomatic_probability = symptomatic_probability_range[1]
        for i in range(6):
            x_beta.append(np.random.triangular(beta_range[0][i], beta_range[1][i], beta_range[2][i],
                                               size=initial_cases))
            x_d_c.append(np.random.triangular(death_range[0][i], death_range[1][i], death_range[2][i],
                                              size=initial_cases))
            x_arrival.append(np.random.triangular(arrival_range[0][i], arrival_range[1][i], arrival_range[2][i],
                                                  size=initial_cases))
            beta.append(beta_range[1][i])
            dc.append(death_range[1][i])
            arrival.append(arrival_range[1][i])
        x_symptomatic = np.random.triangular(symptomatic_probability_range[0], symptomatic_probability_range[1],
                                             symptomatic_probability_range[2], size=initial_cases)

        self.current_results = list()
        manager = multiprocessing.Manager()
        return_list = manager.list()
        jobs = list()
        cores = multiprocessing.cpu_count() - 1
        p = multiprocessing.Process(target=self.calculate_point, args=(real_seroprevalence, real_case, real_death,
                                                                       weights, tuple(beta), tuple(dc), tuple(arrival),
                                                                       symptomatic_probability, dates, return_list,
                                                                       total))
        jobs.append(p)
        for i in range(initial_cases):
            beta = [x_beta[0][i], x_beta[1][i], x_beta[2][i], x_beta[3][i], x_beta[4][i], x_beta[5][i]]
            dc = [x_d_c[0][i], x_d_c[1][i], x_d_c[2][i], x_d_c[3][i], x_d_c[4][i], x_d_c[5][i]]
            arrival = [x_arrival[0][i], x_arrival[1][i], x_arrival[2][i], x_arrival[3][i], x_arrival[4][i],
                       x_arrival[5][i]]
            symptomatic_probability = x_symptomatic[i]
            p = multiprocessing.Process(target=self.calculate_point, args=(real_seroprevalence, real_case, real_death,
                                                                           weights, tuple(beta), tuple(dc),
                                                                           tuple(arrival), symptomatic_probability,
                                                                           dates, return_list, total))
            jobs.append(p)
        block = 0
        for sets in tqdm(chunks(jobs, cores)):
            for job in sets:
                job.start()
            for job in sets:
                job.join()
            for solution in range(max(1, block*cores), min((block+1)*cores-1, len(return_list))):
                jobs2 = list()
                manager2 = multiprocessing.Manager()
                return_list2 = manager2.list()
                v_new = return_list[solution]
                self.current_results.append(v_new)
                if self.ideal_values is None:
                    self.ideal_values = v_new
                elif v_new['error'] < self.ideal_values['error']:
                    self.ideal_values = v_new
                if v_new not in self.results:
                    self.results.append(v_new)
                list_to_add = self.try_new_best(new_point=v_new, real_seroprevalence=real_seroprevalence,
                                                real_case=real_case, real_death=real_death, weights=weights,
                                                dates=dates, result_list=return_list2, total=total)
                for new_point in list_to_add:
                    p = multiprocessing.Process(target=self.calculate_point, args=new_point)
                    jobs2.append(p)
                for job in jobs2:
                    job.start()
                for job in jobs2:
                    job.join()
                for v_new in return_list2:
                    self.current_results.append(v_new)
                    if self.ideal_values is None:
                        self.ideal_values = v_new
                    elif v_new['error'] < self.ideal_values['error']:
                        self.ideal_values = v_new
                    if v_new not in self.results:
                        self.results.append(v_new)
                block += 1

        print('Current best results:')
        for iv in self.ideal_values:
            print(' ', iv, ":", self.ideal_values[iv])
        best_results = pd.DataFrame(self.current_results)
        best_results.drop_duplicates()
        best_results = best_results.sort_values(by='error', ascending=True).head(dimensions+1)
        best_results.reset_index(inplace=True, drop=True)
        best_results = best_results.to_dict(orient='index')
        if self.ideal_values['error'] < min_value_to_iterate:
            it = initial_cases
            n_no_changes = 0
            while n_no_changes < max_no_improvement:
                it += 1
                print('Nelder-Mead iteration', int(it + 1))
                best_error = float(self.ideal_values['error'])
                nelder_mead_result = self.nelder_mead_iteration(best_values=best_results,
                                                                real_seroprevalence=real_seroprevalence,
                                                                real_case=real_case,
                                                                real_death=real_death, pen_weights=weights,
                                                                dates=dates, total=total,
                                                                n_relevant=dimensions)
                best_results_n = nelder_mead_result['values']
                for vi in best_results_n:
                    if best_results_n[vi] not in self.current_results:
                        self.current_results.append(best_results_n[vi])

                max_to_add = len(best_results_n) if self.ideal_values == best_results_n[0] else 5
                for v_test in range(1, max_to_add):
                    jobs2 = list()
                    manager2 = multiprocessing.Manager()
                    return_list = manager2.list()
                    list_to_add = self.try_new_best(new_point=best_results_n[v_test],
                                                    real_seroprevalence=real_seroprevalence,
                                                    real_case=real_case,
                                                    real_death=real_death,
                                                    weights=weights,
                                                    dates=dates,
                                                    result_list=return_list,
                                                    total=total)
                    for new_point in list_to_add:
                        p = multiprocessing.Process(target=self.calculate_point, args=new_point)
                        jobs2.append(p)
                    for sets in chunks(jobs2, cores):
                        for job in sets:
                            job.start()
                        for job in sets:
                            job.join()
                    for v_new in return_list:
                        self.current_results.append(v_new)
                        if self.ideal_values is None:
                            self.ideal_values = v_new
                        elif v_new['error'] < self.ideal_values['error']:
                            self.ideal_values = v_new
                        if v_new not in self.results:
                            self.results.append(v_new)

                best_results = pd.DataFrame(self.current_results).drop_duplicates()
                best_results = best_results.sort_values(by='error', ascending=True).head(dimensions + 1)
                best_results.reset_index(inplace=True, drop=True)
                best_results = best_results.to_dict(orient='index')
                n_no_changes = 0 if round(self.ideal_values['error'], error_precision) < \
                                    round(best_error, error_precision) else n_no_changes + 1
                print('Current best results:')
                for iv in self.ideal_values:
                    print(iv, ":", self.ideal_values[iv])
                print('No improvement in best results in :', n_no_changes, 'iterations')
        else:
            print('Insufficient error value to iterate over NMS: ', self.ideal_values['error'])
        print('Optimum found:')
        print(self.ideal_values)
        organized_results = list()
        for result in self.current_results:
            organized_result = dict()
            current = result
            for i in range(6):
                organized_result['beta_'+str(i)] = current['beta'][i]
            for i in range(6):
                organized_result['dc_' + str(i)] = current['dc'][i]
            for i in range(6):
                organized_result['arrival_' + str(i)] = current['arrival'][i]
            organized_result['error_cases'] = current['error_cases']
            organized_result['error_deaths'] = current['error_deaths']
            organized_result['error'] = current['error']
            organized_results.append(organized_result)
        results_pd = pd.DataFrame(organized_results).drop_duplicates()
        results_pd.reset_index(inplace=True, drop=True)
        file_name = DIR_OUTPUT + 'calibration_nm_results_' + 'W_' + str(weights[0]) + '_' + str(weights[1]) + '_' + \
                    str(weights[2]) + \
                    ' ' + ('total' if total else 'new') + str(iteration)
        with open(file_name + '.json', 'w') as fp_a:
            json.dump(self.current_results, fp_a)
        values = [results_pd.columns] + list(results_pd.values)
        wb = Workbook()
        wb.new_sheet('All_values', data=values)
        wb.save(file_name + '.xlsx')
        print('Excel ', file_name + '.xlsx', 'exported')

        end_processing_s = time.process_time()
        end_time = datetime.datetime.now()
        print('Performance: {0}'.format(end_processing_s - start_processing_s))
        this_time_diff = (end_time - start_time)
        this_execution_time = this_time_diff.total_seconds()
        this_mm = int(this_execution_time / 60)
        this_ss = int(this_execution_time % 60)
        print('Execution Time: {0} minutes {1} seconds'.format(this_mm, this_ss))
        print('Execution Time: {0} milliseconds'.format(this_execution_time * 1000))

    def nelder_mead_iteration(self, best_values: dict, real_seroprevalence: dict, real_case: np.array,
                              real_death: np.array, pen_weights: tuple, dates: dict, n_relevant: int,
                              alpha: float = 1.0, beta_p: float = 0.5, gamma: float = 2.0, delta: float = 0.5,
                              total: bool = True):
        weights = list()
        worst_point = best_values[n_relevant]
        np_worst_point = np.array([worst_point['beta'], worst_point['dc'], worst_point['arrival'], np.ones(6) *
                                   worst_point['spc']])
        for vi in range(len(best_values)-1):
            weights.append(float(abs(best_values[vi]['error']-worst_point['error']) /
                           (sum(sum((best_values[vi][var][i]-worst_point[var][i])**2 for i in range(6))
                                for var in ['beta', 'dc', 'arrival']) +
                            (best_values[vi]['spc']-worst_point['spc'])**2)**0.5)
                           )

        if sum(weights) == 0:
            for vi in range(len(best_values)-1):
                weights[vi] = 1

        centroid = {'beta': [], 'dc': [], 'arrival': [], 'spc': []}
        spc = float(sum(best_values[vi]['spc'] * weights[vi] for vi in range(n_relevant)) / sum(weights))
        for i in range(6):
            centroid['beta'].append(sum(best_values[vi]['beta'][i]*weights[vi] for vi in range(n_relevant)) /
                                    sum(weights))
            centroid['dc'].append(sum(best_values[vi]['dc'][i] * weights[vi] for vi in range(n_relevant)) /
                                  sum(weights))
            centroid['arrival'].append(sum(best_values[vi]['arrival'][i] * weights[vi]
                                           for vi in range(n_relevant)) / sum(weights))
            centroid['spc'].append(spc)
        np_centroid = np.array(list(centroid.values()))
        print('Considered centroid')
        print(centroid)
        manager = multiprocessing.Manager()
        return_list = manager.list()
        jobs = list()

        reflection_point = np.maximum(np_centroid - alpha*(np_centroid-np_worst_point), 0.0)
        beta = tuple(reflection_point[0, :].tolist())
        dc = tuple(reflection_point[1, :].tolist())
        arrival = tuple(reflection_point[2, :].tolist())
        spc = float(reflection_point[3, 0])
        calculate = True
        for point in self.current_results:
            if point['beta'] == beta and point['dc'] == dc and point['arrival'] == arrival and point['spc'] == spc:
                calculate = False
        if calculate:
            p = multiprocessing.Process(target=self.calculate_point, args=(real_seroprevalence, real_case, real_death,
                                                                           pen_weights, tuple(beta), tuple(dc),
                                                                           tuple(arrival), spc,
                                                                           dates, return_list, total))
            jobs.append(p)
            jobs[len(jobs)-1].start()
        expansion_point = np.maximum(np_centroid - gamma * (np_centroid - np_worst_point), 0.0)
        beta = tuple(expansion_point[0, :].tolist())
        dc = tuple(expansion_point[1, :].tolist())
        arrival = tuple(expansion_point[2, :].tolist())
        spc = float(expansion_point[3, 0])
        calculate = True
        for point in self.current_results:
            if point['beta'] == beta and point['dc'] == dc and point['arrival'] == arrival and point['spc'] == spc:
                calculate = False
        if calculate:
            p = multiprocessing.Process(target=self.calculate_point, args=(real_seroprevalence, real_case, real_death,
                                                                           pen_weights, tuple(beta), tuple(dc),
                                                                           tuple(arrival), spc,
                                                                           dates, return_list, total))
            jobs.append(p)
            jobs[len(jobs)-1].start()
        outside_contraction_point = np.maximum(np_centroid + beta_p * (np_centroid - np_worst_point), 0.0)
        beta = tuple(outside_contraction_point[0, :].tolist())
        dc = tuple(outside_contraction_point[1, :].tolist())
        arrival = tuple(outside_contraction_point[2, :].tolist())
        spc = float(outside_contraction_point[3, 0])
        calculate = True
        for point in self.current_results:
            if point['beta'] == beta and point['dc'] == dc and point['arrival'] == arrival and point['spc'] == spc:
                calculate = False
        if calculate:
            p = multiprocessing.Process(target=self.calculate_point, args=(real_seroprevalence, real_case, real_death,
                                                                           pen_weights, tuple(beta), tuple(dc),
                                                                           tuple(arrival), spc,
                                                                           dates, return_list, total))
            jobs.append(p)
            jobs[len(jobs)-1].start()
        inside_contraction_point = np.maximum(np_centroid + beta_p * (np_centroid - np_worst_point), 0.0)
        beta = tuple(inside_contraction_point[0, :].tolist())
        dc = tuple(inside_contraction_point[1, :].tolist())
        arrival = tuple(inside_contraction_point[2, :].tolist())
        spc = float(inside_contraction_point[3, 0])
        calculate = True
        for point in self.current_results:
            if point['beta'] == beta and point['dc'] == dc and point['arrival'] == arrival and point['spc'] == spc:
                calculate = False
        if calculate:
            p = multiprocessing.Process(target=self.calculate_point, args=(real_seroprevalence, real_case, real_death,
                                                                           pen_weights, tuple(beta), tuple(dc),
                                                                           tuple(arrival), spc,
                                                                           dates, return_list, total))
            jobs.append(p)
            jobs[len(jobs)-1].start()
        for j in jobs:
            j.join()
        shrink = True
        for job in return_list:
            self.current_results.append(job)
            self.results.append(job)
            if job['error'] < best_values[n_relevant-1]['error']:
                shrink = False
        if shrink:
            print('Shrink')
            shrink_list = self.calculate_shrinks(values_list=best_values, n_relevant=n_relevant,
                                                 real_seroprevalence=real_seroprevalence,
                                                 real_case=real_case, real_death=real_death, weights=pen_weights,
                                                 dates=dates, total=total, delta=delta)
            for i in range(n_relevant):
                best_values[i + 1] = shrink_list[i]
        else:
            for job in return_list:
                best_values[len(best_values)] = job
        best_values = pd.DataFrame(best_values).T.drop_duplicates()
        best_values = best_values.sort_values(by='error', ascending=True).head(n_relevant + 1)
        best_values.reset_index(inplace=True, drop=True)
        best_values = best_values.to_dict(orient='index')
        return {'values': best_values, 'shrink': shrink}

    def calculate_shrinks(self, values_list: list, n_relevant: int, real_seroprevalence: dict, real_case: np.array,
                          real_death: np.array, dates: dict, weights: tuple, total: bool = True, delta: float = 0.5):
        np_best_point = np.array([values_list[0]['beta'], values_list[0]['dc'], values_list[0]['arrival'],
                                  np.ones(6) * values_list[0]['spc']])
        manager = multiprocessing.Manager()
        return_list = manager.list()
        jobs = list()
        cores = multiprocessing.cpu_count()
        for i in range(n_relevant):
            np_point = np.array([values_list[i + 1]['beta'], values_list[i + 1]['dc'], values_list[i + 1]['arrival'],
                                 np.ones(6) * values_list[i + 1]['spc']])
            shrink_point = np.maximum(np_point + delta * (np_point - np_best_point), 0.0)
            beta = tuple(shrink_point[0, :].tolist())
            dc = tuple(shrink_point[1, :].tolist())
            arrival = tuple(shrink_point[2, :].tolist())
            spc = float(shrink_point[3, 0])
            p = multiprocessing.Process(target=self.calculate_point, args=(real_seroprevalence, real_case, real_death,
                                                                           weights, tuple(beta), tuple(dc),
                                                                           tuple(arrival), spc,
                                                                           dates, return_list, total))
            if len(jobs) < cores:
                jobs.append(p)
                jobs[len(jobs)-1].start()
                time.sleep(2)
            else:
                available = False
                while not available:
                    n_count = 0
                    for j in range(len(jobs)):
                        if jobs[j].is_alive():
                            n_count += 1
                    if n_count < cores:
                        jobs.append(p)
                        available = True
                        jobs[len(jobs)-1].start()
                        time.sleep(2)
                    else:
                        time.sleep(1)
        available = False
        while not available:
            n_count = 0
            for j in jobs:
                if j.is_alive():
                    n_count += 1
            if n_count == 0:
                available = True
            else:
                time.sleep(1)

        return return_list
