from model_run import Model
import numpy as np
import pandas as pd
import datetime
import json
import time
from pyexcelerate import Workbook


class Calibration(object):
    def __init__(self):
        self.model = Model()
        self.ideal_values = dict()
        self.results = list()
        self.type_params = dict()
        for pv in self.model.time_params:
            self.type_params[pv] = 'BASE_VALUE'
        for pv in self.model.prob_params:
            self.type_params[pv] = 'BASE_VALUE'
        self.real_cases = pd.read_csv('input\\real_cases.csv', sep=';')

    @staticmethod
    def __obtain_thetas__(x: np.array, y: np.array):
        try:
            x_matrix = np.column_stack((np.power(x, 2), x, np.ones(x.shape[0])))
            return np.dot(np.linalg.inv(np.dot(np.transpose(x_matrix), x_matrix)), np.dot(np.transpose(x_matrix), y))
        except Exception as e:
            print('Error obtain_thetas: {0}'.format(e))
            return dict()

    def run_calibration(self, initial_cases: int = 30, beta_inf: float = 0.0, beta_base: float = 0.05,
                        beta_sup: float = 1.0, death_inf: float = 0.0, death_base: float = 1, death_sup: float = 2.6,
                        total: bool = True, iteration: int = 1):
        start_processing_s = time.process_time()
        start_time = datetime.datetime.now()

        real_case = np.transpose(np.array(([self.real_cases['cases_total'], self.real_cases['deaths_total']] if total
                                          else [self.real_cases['cases_new'], self.real_cases['deaths_new']]),
                                          dtype='float64'))
        real_case[real_case == 0] = 0.01
        x = np.random.triangular(beta_inf, beta_base, beta_sup, size=initial_cases)
        x2 = np.random.triangular(death_inf, death_base, death_sup, size=initial_cases)
        i = 0
        print('Initial iteration', int(i + 1), 'Beta', beta_base, 'DC', death_base)
        sim_results = self.model.run(self.type_params, name='Calibration' + str(i), run_type='calibration',
                                     beta=beta_base, death_coefficient=death_base, calculated_arrival=True,
                                     sim_length=236)[14:237]
        if not total:
            sim_results_alt = sim_results.copy()
            for k in range(1, len(sim_results)):
                sim_results[k] = sim_results_alt[k] - sim_results_alt[k - 1]
            del sim_results_alt
        y = float(np.average(np.power(sim_results[0:, 0] / real_case[0:, 0] - 1, 2)) +
                  np.average(np.power(sim_results[29:, 1] / real_case[29:, 1] - 1, 2)))
        print('Error:', y)
        v_1 = {'Beta': x[i], 'DC': x2[i], 'Error': y}
        self.results.append(v_1)
        print('Current best results: ', v_1)
        i = 1
        print('Initial iteration', int(i + 1), 'Beta', x[i], 'DC', x2[i])
        sim_results = self.model.run(self.type_params, name='Calibration' + str(i), run_type='calibration', beta=x[i],
                                     death_coefficient=x2[i], calculated_arrival=True, sim_length=236)[14:237]
        if not total:
            sim_results_alt = sim_results.copy()
            for k in range(1, len(sim_results)):
                sim_results[k] = sim_results_alt[k] - sim_results_alt[k - 1]
            del sim_results_alt
        y = float(np.average(np.power(sim_results[0:, 0] / real_case[0:, 0] - 1, 2)) +
                  np.average(np.power(sim_results[29:, 1] / real_case[29:, 1] - 1, 2)))
        print('Error:', y)
        v_new = {'Beta': x[i], 'DC': x2[i], 'Error': y}
        self.results.append(v_new)
        if v_new['Error'] < v_1['Error']:
            v_2 = v_1
            v_1 = v_new
        else:
            v_2 = v_new
        print('Current best results: ', v_1)
        i = 2
        print('Initial iteration', int(i + 1), 'Beta', x[i], 'DC', x2[i])
        sim_results = self.model.run(self.type_params, name='Calibration' + str(i), run_type='calibration', beta=x[i],
                                     death_coefficient=x2[i], calculated_arrival=True, sim_length=236)[14:237]
        if not total:
            sim_results_alt = sim_results.copy()
            for k in range(1, len(sim_results)):
                sim_results[k] = sim_results_alt[k] - sim_results_alt[k - 1]
            del sim_results_alt
        y = float(np.average(np.power(sim_results[0:, 0] / real_case[0:, 0] - 1, 2)) +
                  np.average(np.power(sim_results[29:, 1] / real_case[29:, 1] - 1, 2)))
        print('Error:', y)
        v_new = {'Beta': x[i], 'DC': x2[i], 'Error': y}
        self.results.append(v_new)
        if v_new['Error'] < v_1['Error']:
            v_3 = v_2
            v_2 = v_1
            v_1 = v_new
        elif v_new['Error'] < v_2['Error']:
            v_3 = v_2
            v_2 = v_new
        else:
            v_3 = v_new
        print('Current best results: ', v_1)
        for i in range(3, initial_cases):
            print('Initial iteration', int(i + 1), 'Beta', x[i], 'DC', x2[i])
            sim_results = self.model.run(self.type_params, name='Calibration' + str(i), run_type='calibration',
                                         beta=x[i], death_coefficient=x2[i], calculated_arrival=True, sim_length=236)[14:237]
            if not total:
                sim_results_alt = sim_results.copy()
                for k in range(1, len(sim_results)):
                    sim_results[k] = sim_results_alt[k] - sim_results_alt[k - 1]
                del sim_results_alt
            y = float(np.average(np.power(sim_results[0:, 0] / real_case[0:, 0] - 1, 2)) +
                      np.average(np.power(sim_results[29:, 1] / real_case[29:, 1] - 1, 2)))
            print('Error:', y)
            v_new = {'Beta': x[i], 'DC': x2[i], 'Error': y}
            self.results.append(v_new)
            if v_new['Error'] < v_1['Error']:
                v_3 = v_2
                v_2 = v_1
                v_1 = v_new
            elif v_new['Error'] < v_2['Error']:
                v_3 = v_2
                v_2 = v_new
            else:
                v_3 = v_new
            print('Current best results: ', v_1)

        i = initial_cases
        n_shrinks = 0
        while n_shrinks < 10 and v_1 != v_2 and v_2 != v_3:
            i += 1
            print('Nelder-Mead iteration', int(i + 1))
            current_best_values = [v_1, v_2, v_3]
            nelder_mead_result = self.nelder_mead_iteration(best_values=current_best_values, real_case=real_case,
                                                            total=total)
            if nelder_mead_result['shrink']:
                n_shrinks += 1
                print('n_shrinks: ', n_shrinks)
            else:
                n_shrinks = 0
            v_1, v_2, v_3 = nelder_mead_result['values']
            if v_1 not in self.results:
                self.results.append(v_1)
            if v_2 not in self.results:
                self.results.append(v_2)
            if v_3 not in self.results:
                self.results.append(v_3)
            print('Current best results: ', v_1)
        self.ideal_values = v_1
        print('Optima found: ', v_1)
        results = {'Best': v_1, 'Values': self.results}
        results_pd = pd.DataFrame(self.results)
        with open('output\\calibration_nm_results_' + ('total' if total else 'new') + str(iteration)+ '.json', 'w') as fp:
            json.dump(results, fp)
        values = [results_pd.columns] + list(results_pd.values)
        wb = Workbook()
        wb.new_sheet('All_values', data=values)
        wb.save('output\\calibration_nm_results2_' + ('total' if total else 'new')  + str(iteration)+ '.xlsx')
        print('Excel ', 'output\\calibration_nm_results2_' + ('total' if total else 'new') + '.xlsx', 'exported')

        end_processing_s = time.process_time()
        end_time = datetime.datetime.now()
        print('Performance: {0}'.format(end_processing_s - start_processing_s))
        time_diff = (end_time - start_time)
        execution_time = time_diff.total_seconds()
        mm = int(execution_time / 60)
        ss = int(execution_time % 60)
        print('Execution Time: {0} minutes {1} seconds'.format(mm, ss))
        print('Execution Time: {0} milliseconds'.format(execution_time * 1000))
        return results

    def nelder_mead_iteration(self, best_values: list, real_case: np.array,  alpha: float = 1.0, beta: float = 0.5,
                              gamma: float = 2.0, delta: float = 0.5, total: bool = True):
        v_1 = best_values[0]
        v_2 = best_values[1]
        v_3 = best_values[2]
        w_1 = abs(v_1['Error']-v_3['Error'])/(((v_1['Beta']-v_3['Beta'])**2 + (v_1['DC']-v_3['DC'])**2)**0.5)
        w_2 = abs(v_2['Error'] - v_3['Error']) / \
              (((v_2['Beta'] - v_3['Beta']) ** 2 + (v_2['DC'] - v_3['DC']) ** 2) ** 0.5)
        worst_point = np.array([v_3['Beta'], v_3['DC']])
        centroid = np.array([(w_1*v_1['Beta']+w_2*v_2['Beta'])/(w_1+w_2), (w_1*v_1['DC']+w_2*v_2['DC'])/(w_1+w_2)])
        print(v_1, v_2, v_3)
        print(centroid)
        shrink = False
        # Reflection
        reflection_point = centroid - alpha*(centroid-worst_point)
        reflection_point[0] = max(reflection_point[0], 0)
        reflection_point[1] = max(reflection_point[1], 0)
        sim_results = self.model.run(self.type_params, name='Calibration', run_type='calibration',
                                     beta=float(reflection_point[0]), death_coefficient=float(reflection_point[1]),
                                     calculated_arrival=True, sim_length=236)[14:237]
        if not total:
            sim_results_alt = sim_results.copy()
            for k in range(1, len(sim_results)):
                sim_results[k, :] = sim_results_alt[k, :] - sim_results_alt[k - 1, :]
            del sim_results_alt
        y = float(np.average(np.power(sim_results[0:, 0] / real_case[0:, 0] - 1, 2)) +
                  np.average(np.power(sim_results[29:, 1] / real_case[29:, 1] - 1, 2)))
        if y < v_1['Error']:
            # Expansion
            expansion_point = centroid - gamma*(centroid-worst_point)
            expansion_point[0] = max(expansion_point[0], 0)
            expansion_point[1] = max(expansion_point[1], 0)
            sim_results = self.model.run(self.type_params, name='Calibration', run_type='calibration',
                                         beta=float(expansion_point[0]), death_coefficient=float(expansion_point[1]),
                                         calculated_arrival=True, sim_length=236)[14:237]
            if not total:
                sim_results_alt = sim_results.copy()
                for k in range(1, len(sim_results)):
                    sim_results[k, :] = sim_results_alt[k, :] - sim_results_alt[k - 1, :]
                del sim_results_alt
            y2 = float(np.average(np.power(sim_results[0:, 0] / real_case[0:, 0] - 1, 2)) +
                       np.average(np.power(sim_results[29:, 1] / real_case[29:, 1] - 1, 2)))
            if y2 < y:
                v_3 = v_1
                v_2 = {'Beta': float(reflection_point[0]), 'DC': float(reflection_point[1]), 'Error': y}
                v_1 = {'Beta': float(expansion_point[0]), 'DC': float(expansion_point[1]), 'Error': y2}
            else:
                v_3 = v_1
                v_2 = {'Beta': float(expansion_point[0]), 'DC': float(expansion_point[1]), 'Error': y2}
                v_1 = {'Beta': float(reflection_point[0]), 'DC': float(reflection_point[1]), 'Error': y}
        elif y < v_2['Error']:
            v_3 = v_2
            v_2 = {'Beta': float(reflection_point[0]), 'DC': float(reflection_point[1]), 'Error': y}
        elif y < v_3['Error']:
            # Outside contraction
            outside_contraction_point = centroid + beta*(centroid-worst_point)
            outside_contraction_point[0] = max(outside_contraction_point[0], 0)
            outside_contraction_point[1] = max(outside_contraction_point[1], 0)
            sim_results = self.model.run(self.type_params, name='Calibration', run_type='calibration',
                                         beta=float(outside_contraction_point[0]),
                                         death_coefficient=float(outside_contraction_point[1]), calculated_arrival=True,
                                         sim_length=236)[14:237]
            if not total:
                sim_results_alt = sim_results.copy()
                for k in range(1, len(sim_results)):
                    sim_results[k, :] = sim_results_alt[k, :] - sim_results_alt[k - 1, :]
                del sim_results_alt
            y3 = float(np.average(np.power(sim_results[0:, 0] / real_case[0:, 0] - 1, 2)) +
                       np.average(np.power(sim_results[29:, 1] / real_case[29:, 1] - 1, 2)))
            if y3 < y:
                if y3 < v_1['Error']:
                    v_3 = v_2
                    v_2 = v_2
                    v_1 = {'Beta': float(outside_contraction_point[0]), 'DC': float(outside_contraction_point[1]),
                           'Error': y3}
                elif y3 < v_2['Error']:
                    v_3 = v_2
                    v_2 = {'Beta': float(outside_contraction_point[0]), 'DC': float(outside_contraction_point[1]),
                           'Error': y3}
                elif y3 < v_3['Error']:
                    v_3 = {'Beta': float(outside_contraction_point[0]), 'DC': float(outside_contraction_point[1]),
                           'Error': y3}
            else:
                # Shrink
                shrink = True
                v_1_vec = np.array([v_1['Beta'], v_1['DC']])
                new_v = np.array([v_2['Beta'], v_2['DC']])
                new_v = new_v + delta*(new_v-v_1_vec)
                new_v[0] = max(new_v[0], 0)
                new_v[1] = max(new_v[1], 0)
                sim_results = self.model.run(self.type_params, name='Calibration', run_type='calibration',
                                             beta=float(new_v[0]), death_coefficient=float(new_v[1]),
                                             calculated_arrival=True, sim_length=236)[14:237]
                if not total:
                    sim_results_alt = sim_results.copy()
                    for k in range(1, len(sim_results)):
                        sim_results[k, :] = sim_results_alt[k, :] - sim_results_alt[k - 1, :]
                    del sim_results_alt
                y_s = float(np.average(np.power(sim_results[0:, 0] / real_case[0:, 0] - 1, 2)) +
                                  np.average(np.power(sim_results[29:, 1] / real_case[29:, 1] - 1, 2)))
                v_2 = {'Beta': float(new_v[0]), 'DC': float(new_v[1]), 'Error': y_s}

                new_v = np.array([v_3['Beta'], v_3['DC']])
                new_v = new_v + delta * (new_v - v_1_vec)
                new_v[0] = max(new_v[0], 0)
                new_v[1] = max(new_v[1], 0)
                sim_results = self.model.run(self.type_params, name='Calibration', run_type='calibration',
                                             beta=float(new_v[0]), death_coefficient=float(new_v[1]),
                                             calculated_arrival=True, sim_length=236)[14:237]
                if not total:
                    sim_results_alt = sim_results.copy()
                    for k in range(1, len(sim_results)):
                        sim_results[k, :] = sim_results_alt[k, :] - sim_results_alt[k - 1, :]
                    del sim_results_alt
                y_s = float(np.average(np.power(sim_results[0:, 0] / real_case[0:, 0] - 1, 2)) +
                                  np.average(np.power(sim_results[29:, 1] / real_case[29:, 1] - 1, 2)))
                v_3 = {'Beta': float(new_v[0]), 'DC': float(new_v[1]), 'Error': y_s}
                if v_3['Error'] < v_2['Error']:
                    v_2_temp = v_3
                    v_3 = v_2
                    v_2 = v_2_temp
                if v_2['Error'] < v_1['Error']:
                    v_1_temp = v_2
                    v_2 = v_1
                    v_1 = v_1_temp
                    if v_3['Error'] < v_2['Error']:
                        v_2_temp = v_3
                        v_3 = v_2
                        v_2 = v_2_temp
        else:
            # Inside contraction point
            inside_contraction_point = centroid - beta * (centroid - worst_point)
            inside_contraction_point[0] = max(inside_contraction_point[0], 0)
            inside_contraction_point[1] = max(inside_contraction_point[1], 0)
            sim_results = self.model.run(self.type_params, name='Calibration', run_type='calibration',
                                         beta=float(inside_contraction_point[0]),
                                         death_coefficient=float(inside_contraction_point[1]), calculated_arrival=True,
                                         sim_length=236)[14:237]
            if not total:
                sim_results_alt = sim_results.copy()
                for k in range(1, len(sim_results)):
                    sim_results[k, :] = sim_results_alt[k, :] - sim_results_alt[k - 1, :]
                del sim_results_alt
            y4 = float(np.average(np.power(sim_results[0:, 0] / real_case[0:, 0] - 1, 2)) +
                       np.average(np.power(sim_results[29:, 1] / real_case[29:, 1] - 1, 2)))
            if y4 < v_1['Error']:
                v_3 = v_2
                v_2 = v_1
                v_1 = {'Beta': float(inside_contraction_point[0]), 'DC': float(inside_contraction_point[1]),
                       'Error': y4}
            elif y4 < v_2['Error']:
                v_3 = v_2
                v_2 = {'Beta': float(inside_contraction_point[0]), 'DC': float(inside_contraction_point[1]),
                       'Error': y4}
            elif y4 < v_3['Error']:
                v_3 = {'Beta': float(inside_contraction_point[0]), 'DC': float(inside_contraction_point[1]),
                       'Error': y4}
            else:
                # Shrink
                shrink = True
                v_1_vec = np.array([v_1['Beta'], v_1['DC']])
                new_v = np.array([v_2['Beta'], v_2['DC']])
                new_v = new_v + delta*(new_v-v_1_vec)
                new_v[0] = max(new_v[0], 0)
                new_v[1] = max(new_v[1], 0)
                sim_results = self.model.run(self.type_params, name='Calibration', run_type='calibration',
                                             beta=float(new_v[0]), death_coefficient=float(new_v[1]),
                                             calculated_arrival=True, sim_length=236)[14:237]
                if not total:
                    sim_results_alt = sim_results.copy()
                    for k in range(1, len(sim_results)):
                        sim_results[k, :] = sim_results_alt[k, :] - sim_results_alt[k - 1, :]
                    del sim_results_alt
                y_s = float(np.average(np.power(sim_results[0:, 0] / real_case[0:, 0] - 1, 2)) +
                            np.average(np.power(sim_results[29:, 1] / real_case[29:, 1] - 1, 2)))
                v_2 = {'Beta': float(new_v[0]), 'DC': float(new_v[1]), 'Error': y_s}

                new_v = np.array([v_3['Beta'], v_3['DC']])
                new_v = new_v + delta * (new_v - v_1_vec)
                sim_results = self.model.run(self.type_params, name='Calibration', run_type='calibration',
                                             beta=float(new_v[0]), death_coefficient=float(new_v[1]),
                                             calculated_arrival=True, sim_length=236)[14:237]
                if not total:
                    sim_results_alt = sim_results.copy()
                    for k in range(1, len(sim_results)):
                        sim_results[k, :] = sim_results_alt[k, :] - sim_results_alt[k - 1, :]
                    del sim_results_alt
                y_s = float(np.average(np.power(sim_results[0:, 0] / real_case[0:, 0] - 1, 2)) +
                            np.average(np.power(sim_results[29:, 1] / real_case[29:, 1] - 1, 2)))
                v_3 = {'Beta': float(new_v[0]), 'DC': float(new_v[1]), 'Error': y_s}
        return {'values': [v_1, v_2, v_3], 'shrink': shrink}


start_processing_s_t = time.process_time()
start_time_t = datetime.datetime.now()

c_beta_inf = 0.0
c_beta_base = 0.1 # 0.008140019590906994
c_beta_sup = 0.5
c_death_inf = 1.4
c_death_base = 1.5 # 1.9830377761558986
c_death_sup = 2.6
calibration_model = Calibration()
c_beta_ant = 0.0
c_death_ant = 0.0
n_iteration = 0
# 'Beta': 0.008140019590906994, 'DC': 1.9830377761558986, 'Error': 0.7050015812639594
n_cases = 150
while c_beta_ant != c_beta_base or c_death_ant != c_death_base:
    c_beta_ant = c_beta_base
    c_death_ant = c_death_base
    n_iteration +=1
    print('Cycle number:', n_iteration)
    r = calibration_model.run_calibration(initial_cases=n_cases, beta_inf=c_beta_inf, beta_base=c_beta_base,
                                          beta_sup=c_beta_sup, death_inf=c_death_inf, death_base=c_death_base,
                                          death_sup=c_death_sup, total=True, iteration = n_iteration)
    c_beta_base = r['Best']['Beta']
    c_death_base = r['Best']['DC']
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
