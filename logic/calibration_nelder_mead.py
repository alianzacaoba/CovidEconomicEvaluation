from logic.model import Model
from pyexcelerate import Workbook
import numpy as np
import pandas as pd
import datetime
import json
import time

from root import DIR_INPUT, DIR_OUTPUT


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
        self.real_cases = pd.read_csv(DIR_INPUT + 'real_cases.csv', sep=';')
        self.real_deaths = pd.read_csv(DIR_INPUT + 'death_cases.csv', sep=';')

    def run_calibration(self, beta_range: list, death_range: list, arrival_range: list, dates: dict,
                        dimensions: int = 18, initial_cases: int = 30, total: bool = True, iteration: int = 1,
                        max_shrinks: int = 5, max_no_improvement: int = 100, min_value_to_iterate: float = 10000.0):
        start_processing_s = time.process_time()
        start_time = datetime.datetime.now()
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
        for i in range(6):
            x_beta.append(np.random.triangular(beta_range[1][i]*0.95, beta_range[1][i], beta_range[1][i]*1.05,
                                               size=initial_cases))
            x_d_c.append(np.random.triangular(death_range[1][i]*.95, death_range[1][i], death_range[1][i]*1.05,
                                               size=initial_cases))
            x_arrival.append(np.random.triangular(arrival_range[1][i]*.95, arrival_range[1][i],
                                                  arrival_range[1][i]*1.05, size=initial_cases))
            beta.append(beta_range[1][i])
            dc.append(death_range[1][i])
            arrival.append(arrival_range[1][i])

        print('Initial iteration', int(1))
        print('Parameters \n Beta:', beta, '\n DC:', dc, '\n Arrivals:', arrival)
        sim_results = self.model.run(self.type_params, name='Calibration1', run_type='calibration',
                                     beta=beta, death_coefficient=dc, arrival_coefficient=arrival,
                                     sim_length=236)
        if not total:
            sim_results_alt = sim_results.copy()
            for k in range(1, len(sim_results)):
                print(sim_results[k])
                sim_results[k] = sim_results_alt[k] - sim_results_alt[k - 1]
            del sim_results_alt
        error_cases = float(sum(np.average(np.power(sim_results[dates['days_cases'][i+1]:, i] /
                                              real_case[dates['days_cases'][i+1]:, i]-1, 2)) for i in range(6)))
        error_deaths = float(sum(np.average(np.power(sim_results[dates['days_deaths'][i+1]:, 6+i] /
                                              real_death[dates['days_deaths'][i+1]:, i] - 1, 2)) for i in range(6)))
        error = float(error_cases + error_deaths)
        print('Cases error:', error_cases, 'Deaths error:', error_deaths, 'Total error:', error)
        v_new = {'beta': beta, 'dc': dc, 'arrival': arrival, 'error_cases': error_cases,
                 'error_deaths': error_deaths, 'error': error}
        if v_new not in self.results:
            self.results.append(v_new)
        self.current_results = list()
        self.current_results.append(v_new)
        if self.ideal_values is None:
            best_error = error
            self.ideal_values = v_new
        elif self.ideal_values['error'] > error:
            best_error = error
            self.ideal_values = v_new
        else:
            best_error = self.ideal_values['error']
        print('Current best results:')
        print(self.ideal_values)
        for i in range(initial_cases):
            print('Initial iteration', int(i + 2))
            beta = [x_beta[0][i], x_beta[1][i], x_beta[2][i], x_beta[3][i], x_beta[4][i], x_beta[5][i]]
            dc = [x_d_c[0][i], x_d_c[1][i], x_d_c[2][i], x_d_c[3][i], x_d_c[4][i], x_d_c[5][i]]
            arrival = [x_arrival[0][i], x_arrival[1][i], x_arrival[2][i], x_arrival[3][i], x_arrival[4][i],
                       x_arrival[5][i]]
            print('Parameters \n Beta:', beta, '\n DC:', dc, '\n Arrivals:', arrival)

            sim_results = self.model.run(self.type_params, name='Calibration' + str(i), run_type='calibration',
                                         beta=beta, death_coefficient=dc, arrival_coefficient=arrival,
                                         sim_length=236)
            if not total:
                sim_results_alt = sim_results.copy()
                for k in range(1, len(sim_results)):
                    sim_results[k] = sim_results_alt[k] - sim_results_alt[k - 1]
                del sim_results_alt
            error_cases = float(sum(np.average(np.power(sim_results[dates['days_cases'][i+1]:, i] /
                                                        real_case[dates['days_cases'][i+1]:, i] - 1, 2)) for i in
                                    range(6)))
            error_deaths = float(sum(np.average(np.power(sim_results[dates['days_deaths'][i+1]:, 6 + i] /
                                                         real_death[dates['days_deaths'][i+1]:, i] - 1, 2)) for i in
                                     range(6)))
            error = float(error_cases + error_deaths)
            print('Cases error:', error_cases, 'Deaths error:', error_deaths, 'Total error:', error)
            v_new = {'beta': beta, 'dc': dc, 'arrival': arrival, 'error_cases': error_cases,
                     'error_deaths': error_deaths, 'error': error}
            if v_new not in self.results:
                self.results.append(v_new)
            if v_new not in self.current_results:
                self.current_results.append(v_new)

            if best_error > error:
                best_error = error
                self.ideal_values = v_new
            else:
                best_error = self.ideal_values['error']
            print('Current best results:')
            print(self.ideal_values)

        best_results = pd.DataFrame(self.current_results)
        best_results = best_results.sort_values(by='error', ascending=True, ignore_index=True).head(dimensions+1)
        best_results = best_results.to_dict(orient='index')

        if self.ideal_values['error'] < min_value_to_iterate:
            i = initial_cases
            n_shrinks = 0
            n_no_changes = 0
            while n_shrinks < max_shrinks and n_no_changes < max_no_improvement:
                i += 1
                print('Nelder-Mead iteration', int(i + 1))
                nelder_mead_result = self.nelder_mead_iteration(best_values=best_results, real_case=real_case,
                                                                real_death=real_death, dates=dates, total=total,
                                                                n_relevant=dimensions)
                if nelder_mead_result['shrink']:
                    n_shrinks += 1
                    print('n_shrinks: ', n_shrinks)
                else:
                    n_shrinks = 0
                best_results_n = nelder_mead_result['values']
                for vi in best_results_n:
                    if best_results_n[vi] not in self.current_results:
                        self.current_results.append(best_results_n[vi])
                equal = True
                for vi in range(5):
                    if best_results_n[vi] != best_results[vi]:
                        equal = False
                n_no_changes = n_no_changes + 1 if equal else 0
                best_results = best_results_n
                self.ideal_values = best_results[0]
                print('Current best results:')
                print(best_results)
                print('No improvement in top 5 results in :', n_no_changes, 'iterations')
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
        results_pd = pd.DataFrame(organized_results)
        with open(DIR_OUTPUT + 'calibration_nm_results_' + ('total' if total else 'new') + str(iteration) + '.json',
                  'w') as fp_a:
            json.dump(self.current_results, fp_a)
        values = [results_pd.columns] + list(results_pd.values)
        wb = Workbook()
        wb.new_sheet('All_values', data=values)
        wb.save(DIR_OUTPUT + 'calibration_nm_results_' + ('total' if total else 'new') + str(iteration) + '.xlsx')
        print('Excel ', DIR_OUTPUT + 'calibration_nm_results_2_' + ('total' if total else 'new') + '.xlsx', 'exported')

        end_processing_s = time.process_time()
        end_time = datetime.datetime.now()
        print('Performance: {0}'.format(end_processing_s - start_processing_s))
        this_time_diff = (end_time - start_time)
        this_execution_time = this_time_diff.total_seconds()
        this_mm = int(this_execution_time / 60)
        this_ss = int(this_execution_time % 60)
        print('Execution Time: {0} minutes {1} seconds'.format(this_mm, this_ss))
        print('Execution Time: {0} milliseconds'.format(this_execution_time * 1000))

    def nelder_mead_iteration(self, best_values: dict, real_case: np.array,  real_death: np.array, dates: dict,
                              n_relevant: int, alpha: float = 1.0, beta_p: float = 0.5, gamma: float = 2.0,
                              delta: float = 0.5, total: bool = True):
        weights = list()
        worst_point = best_values[n_relevant]
        np_worst_point = np.array([worst_point['beta'], worst_point['dc'], worst_point['arrival']])
        for vi in range(len(best_values)-1):
            weights.append(float(abs(best_values[vi]['error']-worst_point['error']) /
                           (sum(sum((best_values[vi][var][i]-worst_point[var][i])**2 for i in range(6))
                             for var in ['beta', 'dc', 'arrival'])**0.5)))
        centroid = {'beta': [], 'dc': [], 'arrival': []}
        for i in range(6):
            centroid['beta'].append(sum(best_values[vi]['beta'][i]*weights[vi] for vi in range(n_relevant)) /
                                    sum(weights))
            centroid['dc'].append(sum(best_values[vi]['dc'][i] * weights[vi] for vi in range(n_relevant)) /
                                  sum(weights))
            centroid['arrival'].append(sum(best_values[vi]['arrival'][i] * weights[vi]
                                           for vi in range(n_relevant)) / sum(weights))
        np_centroid = np.array(list(centroid.values()))
        print('Considered centroid')
        print(centroid)
        shrink = False
        reflection_point = np.maximum(np_centroid - alpha*(np_centroid-np_worst_point), 0.0)
        beta = reflection_point[0, :].tolist()
        dc = reflection_point[0, :].tolist()
        arrival = reflection_point[0, :].tolist()
        sim_results = self.model.run(self.type_params, name='Calibration1', run_type='calibration', beta=beta,
                                     death_coefficient=dc, arrival_coefficient=arrival, sim_length=236)
        if not total:
            sim_results_alt = sim_results.copy()
            for k in range(1, len(sim_results)):
                print(sim_results[k])
                sim_results[k] = sim_results_alt[k] - sim_results_alt[k - 1]
            del sim_results_alt
        error_cases = float(sum(np.average(np.power(sim_results[dates['days_cases'][i+1]:, i] /
                                                    real_case[dates['days_cases'][i+1]:, i] - 1, 2)) for i in range(6)))
        error_deaths = float(sum(np.average(np.power(sim_results[dates['days_deaths'][i+1]:, 6 + i] /
                                                     real_death[dates['days_deaths'][i+1]:, i] - 1, 2)) for i in
                                 range(6)))
        error = float(error_cases + error_deaths)
        v_reflection = {'beta': beta, 'dc': dc, 'arrival': arrival, 'error_cases': error_cases,
                 'error_deaths': error_deaths, 'error': error}

        if v_reflection not in self.results:
            self.results.append(v_reflection)
        if v_reflection['error'] < best_values[0]['error']:
            exists = False
            for vi in best_values:
                if v_reflection == best_values[vi]:
                    exists = True
            if not exists:
                best_values[len(best_values)] = v_reflection
            expansion_point = np.maximum(np_centroid - gamma*(np_centroid-np_worst_point), 0.0)
            beta = expansion_point[0, :].tolist()
            dc = expansion_point[0, :].tolist()
            arrival = expansion_point[0, :].tolist()
            sim_results = self.model.run(self.type_params, name='Calibration1', run_type='calibration', beta=beta,
                                         death_coefficient=dc, arrival_coefficient=arrival, sim_length=236)
            if not total:
                sim_results_alt = sim_results.copy()
                for k in range(1, len(sim_results)):
                    print(sim_results[k])
                    sim_results[k] = sim_results_alt[k] - sim_results_alt[k - 1]
                del sim_results_alt
            error_cases = float(sum(np.average(np.power(sim_results[dates['days_cases'][i+1]:, i] /
                                                        real_case[dates['days_cases'][i+1]:, i] - 1, 2)) for i in
                                    range(6)))
            error_deaths = float(sum(np.average(np.power(sim_results[dates['days_deaths'][i+1]:, 6 + i] /
                                                         real_death[dates['days_deaths'][i+1]:, i] - 1, 2)) for i in
                                     range(6)))
            error = float(error_cases + error_deaths)
            v_expansion = {'beta': beta, 'dc': dc, 'arrival': arrival, 'error_cases': error_cases,
                            'error_deaths': error_deaths, 'error': error}
            if v_expansion not in self.results:
                self.results.append(v_expansion)
            if v_expansion['error'] < v_reflection['error']:
                exists = False
                for vi in best_values:
                    if v_expansion == best_values[vi]:
                        exists = True
                if not exists:
                    best_values[len(best_values)] = v_expansion
        elif v_reflection['error'] < best_values[1]['error']:
            exists = False
            for vi in best_values:
                if v_reflection == best_values[vi]:
                    exists = True
            if not exists:
                best_values[len(best_values)] = v_reflection
        elif v_reflection['error'] < worst_point['error']:
            # Outside contraction
            outside_contraction_point = np.maximum(np_centroid + beta_p*(np_centroid-np_worst_point), 0.0)
            beta = outside_contraction_point[0, :].tolist()
            dc = outside_contraction_point[0, :].tolist()
            arrival = outside_contraction_point[0, :].tolist()
            sim_results = self.model.run(self.type_params, name='Calibration1', run_type='calibration', beta=beta,
                                         death_coefficient=dc, arrival_coefficient=arrival, sim_length=236)
            if not total:
                sim_results_alt = sim_results.copy()
                for k in range(1, len(sim_results)):
                    print(sim_results[k])
                    sim_results[k] = sim_results_alt[k] - sim_results_alt[k - 1]
                del sim_results_alt
            error_cases = float(sum(np.average(np.power(sim_results[dates['days_cases'][i+1]:, i] /
                                                        real_case[dates['days_cases'][i+1]:, i] - 1, 2)) for i in
                                    range(6)))
            error_deaths = float(sum(np.average(np.power(sim_results[dates['days_deaths'][i+1]:, 6 + i] /
                                                         real_death[dates['days_deaths'][i+1]:, i] - 1, 2)) for i in
                                     range(6)))
            error = float(error_cases + error_deaths)
            v_outside_contraction = {'beta': beta, 'dc': dc, 'arrival': arrival, 'error_cases': error_cases,
                           'error_deaths': error_deaths, 'error': error}

            if v_outside_contraction not in self.results:
                self.results.append(v_outside_contraction)
            if v_outside_contraction['error'] < v_reflection['error']:
                exists = False
                for vi in best_values:
                    if v_outside_contraction == best_values[vi]:
                        exists = True
                if not exists:
                    best_values[len(best_values)] = v_outside_contraction
            else:
                # Shrink
                shrink = True
                np_best_point = np.array([best_values[0]['beta'], best_values[0]['dc'], best_values[0]['arrival']])
                for i in range(n_relevant):
                    np_point = np.array([best_values[i+1]['beta'], best_values[i+1]['dc'], best_values[i+1]['arrival']])
                    shrink_point = np.maximum(np_point + delta*(np_point-np_best_point), 0.0)
                    beta = shrink_point[0, :].tolist()
                    dc = shrink_point[0, :].tolist()
                    arrival = shrink_point[0, :].tolist()
                    sim_results = self.model.run(self.type_params, name='Calibration1', run_type='calibration',
                                                 beta=beta,
                                                 death_coefficient=dc, arrival_coefficient=arrival, sim_length=236)
                    if not total:
                        sim_results_alt = sim_results.copy()
                        for k in range(1, len(sim_results)):
                            print(sim_results[k])
                            sim_results[k] = sim_results_alt[k] - sim_results_alt[k - 1]
                        del sim_results_alt
                    error_cases = float(sum(np.average(np.power(sim_results[dates['days_cases'][i+1]:, i] /
                                                                real_case[dates['days_cases'][i+1]:, i] - 1, 2))
                                            for i in range(6)))
                    error_deaths = float(sum(np.average(np.power(sim_results[dates['days_deaths'][i+1]:, 6 + i] /
                                                                 real_death[dates['days_deaths'][i+1]:, i] - 1, 2))
                                             for i in range(6)))
                    error = float(error_cases + error_deaths)
                    v_shrink = {'beta': beta, 'dc': dc, 'arrival': arrival, 'error_cases': error_cases,
                                             'error_deaths': error_deaths, 'error': error}
                    if v_shrink not in self.results:
                        self.results.append(v_shrink)
                    best_values[i + 1] = v_shrink
        else:
            # Inside contraction point
            inside_contraction_point = np.maximum(np_centroid + beta_p * (np_centroid - np_worst_point), 0.0)
            beta = inside_contraction_point[0, :].tolist()
            dc = inside_contraction_point[0, :].tolist()
            arrival = inside_contraction_point[0, :].tolist()
            sim_results = self.model.run(self.type_params, name='Calibration1', run_type='calibration', beta=beta,
                                         death_coefficient=dc, arrival_coefficient=arrival, sim_length=236)
            if not total:
                sim_results_alt = sim_results.copy()
                for k in range(1, len(sim_results)):
                    print(sim_results[k])
                    sim_results[k] = sim_results_alt[k] - sim_results_alt[k - 1]
                del sim_results_alt
            error_cases = float(sum(np.average(np.power(sim_results[dates['days_cases'][i+1]:, i] /
                                                        real_case[dates['days_cases'][i+1]:, i] - 1, 2)) for i in
                                    range(6)))
            error_deaths = float(sum(np.average(np.power(sim_results[dates['days_deaths'][i+1]:, 6 + i] /
                                                         real_death[dates['days_deaths'][i+1]:, i] - 1, 2)) for i in
                                     range(6)))
            error = float(error_cases + error_deaths)
            v_inside_contraction = {'beta': beta, 'dc': dc, 'arrival': arrival, 'error_cases': error_cases,
                                     'error_deaths': error_deaths, 'error': error}

            if v_inside_contraction not in self.results:
                self.results.append(v_inside_contraction)
            if v_inside_contraction['error'] < worst_point['error']:
                exists = False
                for vi in best_values:
                    if v_inside_contraction == best_values[vi]:
                        exists = True
                if not exists:
                    best_values[len(best_values)] = v_inside_contraction
            else:
                # Shrink
                shrink = True
                np_best_point = np.array([best_values[0]['beta'], best_values[0]['dc'], best_values[0]['arrival']])
                for i in range(n_relevant):
                    np_point = np.array(
                        [best_values[i + 1]['beta'], best_values[i + 1]['dc'], best_values[i + 1]['arrival']])
                    shrink_point = np.maximum(np_point + delta * (np_point - np_best_point), 0)
                    beta = shrink_point[0, :].tolist()
                    dc = shrink_point[0, :].tolist()
                    arrival = shrink_point[0, :].tolist()
                    sim_results = self.model.run(self.type_params, name='Calibration1', run_type='calibration',
                                                 beta=beta,
                                                 death_coefficient=dc, arrival_coefficient=arrival, sim_length=236)
                    if not total:
                        sim_results_alt = sim_results.copy()
                        for k in range(1, len(sim_results)):
                            print(sim_results[k])
                            sim_results[k] = sim_results_alt[k] - sim_results_alt[k - 1]
                        del sim_results_alt
                    error_cases = float(sum(np.average(np.power(sim_results[dates['days_cases'][i+1]:, i] /
                                                                real_case[dates['days_cases'][i+1]:, i] - 1, 2))
                                            for i in range(6)))
                    error_deaths = float(sum(np.average(np.power(sim_results[dates['days_deaths'][i+1]:, 6 + i] /
                                                                 real_death[dates['days_deaths'][i+1]:, i] - 1, 2))
                                             for i in range(6)))
                    error = float(error_cases + error_deaths)
                    v_shrink = {'beta': beta, 'dc': dc, 'arrival': arrival, 'error_cases': error_cases,
                                'error_deaths': error_deaths, 'error': error}
                    if v_shrink not in self.results:
                        self.results.append(v_shrink)
                    best_values[i + 1] = v_shrink
        best_values = pd.DataFrame(best_values).T
        best_values = best_values.sort_values(by='error', ascending=True, ignore_index=True).head(n_relevant + 1)
        best_values = best_values.to_dict(orient='index')
        return {'values': best_values, 'shrink': shrink}