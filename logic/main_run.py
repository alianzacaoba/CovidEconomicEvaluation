import multiprocessing
from logic.model import Model
import pandas as pd
import time
import datetime
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from root import DIR_INPUT, DIR_OUTPUT

matplotlib.use('Agg')

class MainRun(object):

    def __init__(self):
        self.model_ex = Model()
        self.type_paramsA = dict()
        for pv in self.model_ex.time_params:
            self.type_paramsA[pv[0]] = 'BASE_VALUE'
        for pv in self.model_ex.prob_params:
            self.type_paramsA[pv[0]] = 'BASE_VALUE'

        priority_scenarios = pd.read_csv(DIR_INPUT + 'priority_vaccines.csv')
        self.priority_vaccine_scenarios = dict()
        for sc in priority_scenarios.SCENARIO.unique():
            scenario = priority_scenarios[priority_scenarios.SCENARIO == sc][['PHASE', 'AGE_GROUP', 'WORK_GROUP',
                                                                              'HEALTH_GROUP', 'PERCENT']].sort_values(
                by='PHASE', ascending=True)
            phases = list()
            for ph in scenario.PHASE.unique():
                phases.append(scenario[scenario.PHASE == ph][['AGE_GROUP', 'WORK_GROUP', 'HEALTH_GROUP',
                                                              'PERCENT']].values.tolist())
            self.priority_vaccine_scenarios[sc] = phases
        vaccine_effectiveness_df = pd.read_csv(DIR_INPUT + 'vaccine_effectiveness.csv', sep=';')
        self.vaccine_effectiveness_scenarios = dict()
        for index, row in vaccine_effectiveness_df.iterrows():
            self.vaccine_effectiveness_scenarios[row['SCENARIO']] = \
                self.vaccine_effectiveness_scenarios.get(row['SCENARIO'], dict())
            self.vaccine_effectiveness_scenarios[row['SCENARIO']][(row['AGE_GROUP'], row['HEALTH_GROUP'])] = \
                {'VACCINE_EFFECTIVENESS_1': row['VACCINE_EFFECTIVENESS_1'],
                 'VACCINE_EFFECTIVENESS_2': row['VACCINE_EFFECTIVENESS_2']}
        del vaccine_effectiveness_df

        self.vaccine_start_days = {'INF_VALUE': 377, 'BASE_VALUE': 419, 'MAX_VALUE': 480}
        self.vaccine_end_days = {'INF_VALUE': 682, 'BASE_VALUE': 724, 'MAX_VALUE': 785}
        self.n_vaccine_days = 305
        self.type_paramsA['daly'] = 'BASE_VALUE'
        self.type_paramsA['cost'] = 'BASE_VALUE'
        self.type_paramsA['vaccine_day'] = 'BASE_VALUE'
        self.vaccine_information = pd.read_csv(DIR_INPUT + 'region_capacities.csv', sep=';', index_col=0).to_dict()

    def run(self, c_beta_base: tuple, c_death_base: tuple, c_arrival_base: tuple, spc: float):
        jobs = list()
        cores = multiprocessing.cpu_count() - 1
        p = multiprocessing.Process(target=self.model_ex.run, args=(self.type_paramsA, 'no_vac', 'no_vaccination',
                                                                    c_beta_base,
                                                               c_death_base, c_arrival_base, spc, None, None, None,
                                                                    None, None,
                                                               (365 * 3), 'csv', False))
        jobs.append(p)
        jobs[0].start()
        for pvs in self.priority_vaccine_scenarios:
            for pes in self.vaccine_effectiveness_scenarios:
                c_name = 'vac_priority_' + pvs + '_effectiveness_' + str(pes)
                p = multiprocessing.Process(target=self.model_ex.run,
                                            args=(self.type_paramsA, c_name, 'no_vaccination', c_beta_base,
                                                  c_death_base, c_arrival_base, spc,
                                                  self.priority_vaccine_scenarios[pvs],
                                                  self.vaccine_information['VACCINE_CAPACITY'],
                                                  self.vaccine_effectiveness_scenarios[pes], self.vaccine_start_days,
                                                  self.vaccine_end_days, (365 * 3), 'csv', False))
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

        for pv in self.type_paramsA:
            type_params_b = self.type_paramsA.copy()
            for val in ['INF_VALUE', 'MAX_VALUE']:
                type_params_b[pv] = val
                c_name = 'sensitivity_' + pv + '_' + val + '_vac_priority_no_vac'
                p = multiprocessing.Process(target=self.model_ex.run,
                                            args=(type_params_b, c_name, 'no_vaccination', c_beta_base, c_death_base,
                                                  c_arrival_base, spc, None, None, None, None, None, (365 * 3), 'csv',
                                                  False))
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

                for pvs in self.priority_vaccine_scenarios:
                    for pes in self.vaccine_effectiveness_scenarios:
                        c_name = 'sensitivity_' + pv + '_' + val + '_vac_priority_' + pvs + '_effectiveness_' + \
                                 str(pes)
                        p = multiprocessing.Process(target=self.model_ex.run,
                                                    args=(type_params_b, c_name, 'no_vaccination', c_beta_base,
                                                          c_death_base, c_arrival_base, spc,
                                                          self.priority_vaccine_scenarios[pvs],
                                                          self.vaccine_information['VACCINE_CAPACITY'],
                                                          self.vaccine_effectiveness_scenarios[pes],
                                                          self.vaccine_start_days, self.vaccine_end_days, (365 * 3),
                                                          'csv', False))
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

    def run_quality_test(self, c_beta_base: tuple, c_death_base: tuple, c_arrival_base: tuple, spc: float,
                         name: str = 'no vac'):
        self.model_ex.run(self.type_paramsA, name, 'no_vaccination', c_beta_base, c_death_base, c_arrival_base, spc,
                          None, None, None, None, None, (365 * 3), 'csv', False)
        df = pd.read_csv(DIR_OUTPUT + 'result_' + name + '.csv')
        df['Date'] = df['day'].apply(lambda x: datetime.datetime(2020, 2, 21) + datetime.timedelta(days=x))
        df.drop(columns='day', inplace=True)
        df2 = df[['Date', 'Department', 'Health', 'Work', 'Age', 'Susceptible', 'Exposed', 'Presymptomatic',
                  'Symptomatic', 'Home', 'Hospitalization', 'ICU', 'In_recovery', 'Asymptomatic', 'Recovered', 'Immune',
                  'Death']]
        df2 = df2.drop(columns=['Department', 'Health', 'Work', 'Age']).groupby('Date').sum().reset_index(drop=False)
        df2 = pd.melt(df2, id_vars=['Date'], var_name='Health_State', value_name='Population')
        fig, ax = plt.subplots(figsize=(20, 5))
        sns.set_style('darkgrid')
        sns.lineplot(data=df2, x='Date', y='Population', hue='Health_State', style='Health_State', ax=ax)
        plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
        fig.savefig(DIR_OUTPUT + name + '.pdf')
        df_compare = pd.read_csv(DIR_INPUT + 'death_cases.csv')
        df_compare['Real_Deaths'] = \
            df_compare[['TOTAL_1', 'TOTAL_2', 'TOTAL_3', 'TOTAL_4', 'TOTAL_5', 'TOTAL_6']].sum(axis=1)
        df_compare.rename(columns={'DATE': 'Date'}, inplace=True)
        df_compare.Date = pd.to_datetime(df_compare.Date)
        df_compare.set_index('Date', inplace=True)
        df_real = pd.read_csv(DIR_INPUT + 'real_cases.csv')
        df_real.rename(columns={'DATE': 'Date'}, inplace=True)
        df_real['Real_Cases'] = df_real[['TOTAL_1', 'TOTAL_2', 'TOTAL_3', 'TOTAL_4', 'TOTAL_5', 'TOTAL_6']].sum(axis=1)
        df_real.Date = pd.to_datetime(df_real.Date)
        df_real.set_index('Date', inplace=True)
        df_compare = pd.concat([df_compare['Real_Deaths'], df_real['Real_Cases']], axis=1, join='inner')
        df_simulation_compare = df[['Date', 'Cases', 'Death']].groupby(['Date']).sum()
        df_compare = pd.concat([df_compare, df_simulation_compare], axis=1, join='inner').reset_index(drop=False)
        df_compare.rename(columns={'Death': 'Simulated_Deaths', 'Cases': 'Simulated_Cases'}, inplace=True)
        ax = df_compare[['Date', 'Real_Cases', 'Simulated_Cases']].plot(kind='line', x='Date',
                                                                    title='Comparison of Cases Error', figsize=(10, 5))
        ax.set_ylabel('Cases')
        fig = ax.get_figure()
        fig.savefig(DIR_OUTPUT + name + 'CasesError.pdf')

        ax = df_compare[['Date', 'Real_Deaths', 'Simulated_Deaths']].plot(kind='line', x='Date',
                                                                      title='Comparison of Death Error',
                                                                      figsize=(10, 5))
        ax.set_ylabel('Deaths')
        fig = ax.get_figure()
        fig.savefig(DIR_OUTPUT + name + 'DeathError.pdf')
        df_compare[['Date', 'Real_Deaths', 'Simulated_Deaths']].to_csv(DIR_OUTPUT + name + 'DeathComparison.csv',
                                                                       index=False)
        df_compare[['Date', 'Real_Cases', 'Simulated_Cases']].to_csv(DIR_OUTPUT + name + 'CaseComparison.csv',
                                                                     index=False)
