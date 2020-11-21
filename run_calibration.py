from model_run import Model
import numpy as np
import pandas as pd
import datetime
import json
import time


class Calibration(object):
    def __init__(self):
        self.model = Model()
        self.ideal_beta = None
        self.lowest_error = None
        self.results = {'betas': list(), 'errors': list()}
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

    def run_calibration(self, initial_cases: int = 30, inf: float = 0.0, base: float = 0.05, sup: float = 1.0,
                        total: bool = True):
        start_processing_s = time.process_time()
        start_time = datetime.datetime.now()
        real_case = self.real_cases['total'] if total else self.real_cases['new']
        x = np.random.triangular(inf, base, sup, size=initial_cases)
        for i in range(initial_cases):
            x1 = x[i]
            x2 = x[i]*0.9
            x3 = x[i]*1.05
            print('Initial iteration', int(i+1), 'Betas', x1, x2, x3)
            sim_case1 = self.model.run(self.type_params, name='Calibration' + str(i), run_type='calibration',
                                        beta=x1, calculated_arrival=True, sim_length=236)[14:237]
            sim_case2 = self.model.run(self.type_params, name='Calibration' + str(i), run_type='calibration',
                                        beta=x2, calculated_arrival=True, sim_length=236)[14:237]
            sim_case3 = self.model.run(self.type_params, name='Calibration' + str(i), run_type='calibration',
                                        beta=x3, calculated_arrival=True, sim_length=236)[14:237]
            if not total:
                sim_case_alt = sim_case1.copy()
                for k in range(1, len(sim_case1)):
                    sim_case1[k] = sim_case_alt[k] - sim_case_alt[k - 1]
                sim_case_alt = sim_case2.copy()
                for k in range(1, len(sim_case2)):
                    sim_case2[k] = sim_case_alt[k] - sim_case_alt[k - 1]
                sim_case_alt = sim_case3.copy()
                for k in range(1, len(sim_case2)):
                    sim_case3[k] = sim_case_alt[k] - sim_case_alt[k - 1]
                del sim_case_alt

            y1 = float(np.average(np.power(sim_case1 / real_case - 1, 2)))
            self.results['betas'].append(x1)
            self.results['errors'].append(y1)

            y2 = float(np.average(np.power(sim_case2 / real_case - 1, 2)))
            self.results['betas'].append(x2)
            self.results['errors'].append(y2)

            y3 = float(np.average(np.power(sim_case3 / real_case - 1, 2)))
            self.results['betas'].append(x3)
            self.results['errors'].append(y3)
            print('Errors', y1, y2, y3)
        x_old = x[initial_cases-1]
        theta = Calibration.__obtain_thetas__(np.array(self.results['betas']), np.array(self.results['errors']))
        print('Current thetas', theta)
        x = -theta[1] / (2 * theta[0]) if theta[0] != 0 else np.random.triangular(inf, base, sup)
        while x not in self.results['betas']:
            x1 = x
            x2 = max(2*x - x_old, 0)
            x3 = max(2*x_old - x, 0)
            print('Calculated iteration', int(len(self.results['betas'])/3+1), 'Betas', x1, x2, x3,
                  datetime.datetime.now())
            sim_case1 = self.model.run(self.type_params, name='Calibration' + str(x1), run_type='calibration',
                                       beta=x1, calculated_arrival=True, sim_length=236)[14:237]
            sim_case2 = self.model.run(self.type_params, name='Calibration' + str(x2), run_type='calibration',
                                       beta=x2, calculated_arrival=True, sim_length=236)[14:237]
            sim_case3 = self.model.run(self.type_params, name='Calibration' + str(x3), run_type='calibration',
                                       beta=x3, calculated_arrival=True, sim_length=236)[14:237]
            if not total:
                sim_case_alt = sim_case1.copy()
                for k in range(1, len(sim_case1)):
                    sim_case1[k] = sim_case_alt[k] - sim_case_alt[k - 1]
                sim_case_alt = sim_case2.copy()
                for k in range(1, len(sim_case2)):
                    sim_case2[k] = sim_case_alt[k] - sim_case_alt[k - 1]
                sim_case_alt = sim_case3.copy()
                for k in range(1, len(sim_case2)):
                    sim_case3[k] = sim_case_alt[k] - sim_case_alt[k - 1]
                del sim_case_alt

            y1 = float(np.average(np.power(sim_case1 / real_case - 1, 2)))
            self.results['betas'].append(x1)
            self.results['errors'].append(y1)

            y2 = float(np.average(np.power(sim_case2 / real_case - 1, 2)))
            self.results['betas'].append(x2)
            self.results['errors'].append(y2)

            y3 = float(np.average(np.power(sim_case3 / real_case - 1, 2)))
            self.results['betas'].append(x3)
            self.results['errors'].append(y3)
            print('Errors', y1, y2, y3)
            theta = Calibration.__obtain_thetas__(np.array(self.results['betas']), np.array(self.results['errors']))
            print('Current thetas', theta)
            x_old = x
            x = -theta[1] / (2 * theta[0]) if theta[0] != 0 else np.random.triangular(0, base, sup)
        self.lowest_error = min(self.results['errors'])
        self.ideal_beta = self.results['betas'][self.results['errors'].index([self.lowest_error])]
        results = {'beta': self.ideal_beta, 'error': self.lowest_error, 'iterations': self.results}
        print('beta', self.ideal_beta, 'error', self.lowest_error)
        with open('output\\calibration_results_' + ('total' if total else 'new') + '.json', 'w') as fp:
            json.dump(results, fp)

        end_processing_s = time.process_time()
        end_time = datetime.datetime.now()
        print('Performance: {0}'.format(end_processing_s - start_processing_s))
        time_diff = (end_time - start_time)
        execution_time = time_diff.total_seconds()
        mm = int(execution_time / 60)
        ss = int(execution_time % 60)
        print('Execution Time: {0} minutes {1} seconds'.format(mm, ss))
        print('Execution Time: {0} milliseconds'.format(execution_time * 1000))


# calibration_model_2 = Calibration()
# calibration_model_2.run_calibration(initial_cases=30, base=c_base, sup=c_sup, total=False)
calibration_model = Calibration()
c_inf = 0.004
c_base = 0.007
c_sup = 0.008
calibration_model.run_calibration(initial_cases=15, inf=c_inf, base=c_base, sup=c_sup, total=True)
