import multiprocessing

from tqdm import tqdm

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
        self.vaccine_start_days = {'INF_VALUE': 377, 'BASE_VALUE': 419, 'MAX_VALUE': 480}
        self.vaccine_end_days = {'INF_VALUE': 682, 'BASE_VALUE': 724, 'MAX_VALUE': 785}
        self.n_vaccine_days = 305
        self.type_paramsA['daly'] = 'BASE_VALUE'
        self.type_paramsA['cost'] = 'BASE_VALUE'
        self.type_paramsA['vaccine_day'] = 'BASE_VALUE'
        self.vaccine_information = pd.read_excel(DIR_INPUT + 'vaccine_info.xlsx', index_col=0).to_dict('index')

    def run(self, c_beta_base: tuple, c_death_base: tuple, c_arrival_base: tuple, spc: float):

        '''
        type_params: dict, 
        name: str = 'Iteration', 
        run_type: str = 'vaccination',
        beta: tuple = beta_f, 
        death_coefficient: tuple = dc_f, 
        arrival_coefficient: tuple = arrival_f,
        symptomatic_coefficient: float = spc_f, 
        vaccine_priority: list = None, 
        vaccine_information: dict = None,
        vaccine_start_days: dict = None, 
        vaccine_end_days: dict = None, 
        sim_length: int = 365 * 3,
        use_tqdm: bool = False, 
        t_lost_inm: int = 0, 
        n_parallel: int = multiprocessing.cpu_count() - 1,
        exporting_information: Union[str, list] = 'All'
        '''
        c_name = 'no_vac'
        print(c_name, datetime.datetime.now())
        self.model_ex.run(type_params=self.type_paramsA, name=c_name, run_type='no_vaccination',
                              beta=c_beta_base, death_coefficient=c_death_base, arrival_coefficient=c_arrival_base,
                              symptomatic_coefficient=spc, vaccine_priority=None, vaccine_information=None,
                              vaccine_start_days=None, vaccine_end_days=None, sim_length=365 * 3, use_tqdm=True,
                              t_lost_inm=0, n_parallel=34,
                              exporting_information='All')
        for pvs in self.priority_vaccine_scenarios:
            c_name = 'vac_priority_' + pvs
            print(c_name, datetime.datetime.now())
            self.model_ex.run(type_params=self.type_paramsA, name=c_name, run_type='vaccination',
                              beta=c_beta_base, death_coefficient=c_death_base, arrival_coefficient=c_arrival_base,
                              symptomatic_coefficient=spc, vaccine_priority=self.priority_vaccine_scenarios[pvs],
                              vaccine_information=self.vaccine_information,
                              vaccine_start_days=self.vaccine_start_days,
                              vaccine_end_days=self.vaccine_end_days, sim_length=365 * 3, use_tqdm=True,
                              t_lost_inm=0, n_parallel=multiprocessing.cpu_count(),
                              exporting_information='All')
        print('End process')

    def run_immunity_loss_scenarios(self, c_beta_base: tuple, c_death_base: tuple, c_arrival_base: tuple, spc: float):

            '''
            type_params: dict,
            name: str = 'Iteration',
            run_type: str = 'vaccination',
            beta: tuple = beta_f,
            death_coefficient: tuple = dc_f,
            arrival_coefficient: tuple = arrival_f,
            symptomatic_coefficient: float = spc_f,
            vaccine_priority: list = None,
            vaccine_information: dict = None,
            vaccine_start_days: dict = None,
            vaccine_end_days: dict = None,
            sim_length: int = 365 * 3,
            use_tqdm: bool = False,
            t_lost_inm: int = 0,
            n_parallel: int = multiprocessing.cpu_count() - 1,
            exporting_information: Union[str, list] = 'All'
            '''
            for i in range(0, 11):
                c_name = 'no_vac_' + str(i * 5)
                print(c_name, datetime.datetime.now())
                self.model_ex.run(type_params=self.type_paramsA, name=c_name, run_type='no_vaccination',
                                  beta=c_beta_base,
                                  death_coefficient=c_death_base, arrival_coefficient=c_arrival_base,
                                  symptomatic_coefficient=spc, vaccine_priority=None, vaccine_information=None,
                                  vaccine_start_days=None, vaccine_end_days=None, sim_length=365 * 3, use_tqdm=True,
                                  t_lost_inm=i * 5 / (100 * 365), n_parallel=34,
                                  exporting_information=[])
                for pvs in self.priority_vaccine_scenarios:
                    c_name = 'vac_priority_' + pvs + '_' + str(i * 5)
                    print(c_name, datetime.datetime.now())
                    self.model_ex.run(type_params=self.type_paramsA, name=c_name, run_type='vaccination',
                                      beta=c_beta_base, death_coefficient=c_death_base,
                                      arrival_coefficient=c_arrival_base,
                                      symptomatic_coefficient=spc,
                                      vaccine_priority=self.priority_vaccine_scenarios[pvs],
                                      vaccine_information=self.vaccine_information,
                                      vaccine_start_days=self.vaccine_start_days,
                                      vaccine_end_days=self.vaccine_end_days, sim_length=365 * 3, use_tqdm=True,
                                      t_lost_inm=i * 5 / (100 * 365), n_parallel=multiprocessing.cpu_count(),
                                      exporting_information=[])
            print('End process')

    def run_tornado(self, c_beta_base: tuple, c_death_base: tuple, c_arrival_base: tuple, spc: float):

            '''
            type_params: dict,
            name: str = 'Iteration',
            run_type: str = 'vaccination',
            beta: tuple = beta_f,
            death_coefficient: tuple = dc_f,
            arrival_coefficient: tuple = arrival_f,
            symptomatic_coefficient: float = spc_f,
            vaccine_priority: list = None,
            vaccine_information: dict = None,
            vaccine_start_days: dict = None,
            vaccine_end_days: dict = None,
            sim_length: int = 365 * 3,
            use_tqdm: bool = False,
            t_lost_inm: int = 0,
            n_parallel: int = multiprocessing.cpu_count() - 1,
            exporting_information: Union[str, list] = 'All'
            '''

            for pv in self.type_paramsA:
                type_params_b = self.type_paramsA.copy()
                for val in ['INF_VALUE', 'MAX_VALUE']:
                    type_params_b[pv] = val
                    c_name = 'sensitivity_' + pv + '_' + val + '_vac_priority_no_vac'
                    print(c_name, datetime.datetime.now())
                    self.model_ex.run(type_params=type_params_b, name=c_name, run_type='no_vaccination',
                                      beta=c_beta_base,
                                      death_coefficient=c_death_base, arrival_coefficient=c_arrival_base,
                                      symptomatic_coefficient=spc, vaccine_priority=None, vaccine_information=None,
                                      vaccine_start_days=None, vaccine_end_days=None, sim_length=365 * 3, use_tqdm=True,
                                      t_lost_inm=0, n_parallel=34,
                                      exporting_information=[])
                    for pvs in self.priority_vaccine_scenarios:
                        c_name = 'sensitivity_' + pv + '_' + val + '_vac_priority_' + pvs
                        print(c_name, datetime.datetime.now())
                        self.model_ex.run(type_params=type_params_b, name=c_name, run_type='vaccination',
                                          beta=c_beta_base, death_coefficient=c_death_base,
                                          arrival_coefficient=c_arrival_base,
                                          symptomatic_coefficient=spc,
                                          vaccine_priority=self.priority_vaccine_scenarios[pvs],
                                          vaccine_information=self.vaccine_information,
                                          vaccine_start_days=self.vaccine_start_days,
                                          vaccine_end_days=self.vaccine_end_days, sim_length=365 * 3, use_tqdm=True,
                                          t_lost_inm=0, n_parallel=multiprocessing.cpu_count(),
                                          exporting_information=[])
            print('End process')

    def run_quality_test(self, c_beta_base: tuple, c_death_base: tuple, c_arrival_base: tuple, spc: float,
                         name: str = 'no_vac'):
        self.model_ex.run(type_params=self.type_paramsA, name=name, run_type='no_vaccination', beta=c_beta_base,
                          death_coefficient=c_death_base, arrival_coefficient=c_arrival_base,
                          symptomatic_coefficient=spc, sim_length=500, n_parallel=34, use_tqdm=True)
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
        plt.close('all')

    def run_sensibility_spc(self, c_beta_base: tuple, c_death_base: tuple, c_arrival_base: tuple, spc: float):
        for coef in [0.5, 0.75, 1.5, 1.75]:
            c_name = 'no_vac_spc_' + str(coef)
            print(c_name, datetime.datetime.now())
            self.model_ex.run(type_params=self.type_paramsA, name=c_name, run_type='no_vaccination',
                                  beta=c_beta_base, death_coefficient=c_death_base, arrival_coefficient=c_arrival_base,
                                  symptomatic_coefficient=spc*coef, vaccine_priority=None, vaccine_information=None,
                                  vaccine_start_days=None, vaccine_end_days=None, sim_length=365 * 3, use_tqdm=True,
                                  t_lost_inm=0, n_parallel=34,
                                  exporting_information=[])
            for pvs in self.priority_vaccine_scenarios:
                c_name = 'vac_priority_' + pvs + '_spc_' + str(coef)
                print(c_name, datetime.datetime.now())
                self.model_ex.run(type_params=self.type_paramsA, name=c_name, run_type='vaccination',
                                  beta=c_beta_base, death_coefficient=c_death_base, arrival_coefficient=c_arrival_base,
                                  symptomatic_coefficient=spc*coef, vaccine_priority=self.priority_vaccine_scenarios[pvs],
                                  vaccine_information=self.vaccine_information,
                                  vaccine_start_days=self.vaccine_start_days,
                                  vaccine_end_days=self.vaccine_end_days, sim_length=365 * 3, use_tqdm=True,
                                  t_lost_inm=0, n_parallel=multiprocessing.cpu_count(),
                                  exporting_information=[])
        print('End process')

    def run_sensibility_vac_eff(self, c_beta_base: tuple, c_death_base: tuple, c_arrival_base: tuple, spc: float):
        for vac_eff_params in [['R', 0.25], ['R', 0.50], ['V', 0.5], ['V', 0.7], ['V', 0.9]]:
            for pvs in self.priority_vaccine_scenarios:
                c_name = 'vac_priority_' + pvs + '_veff_' + vac_eff_params[0] + '_' + str(vac_eff_params[1])
                print(c_name, datetime.datetime.now())
                self.model_ex.run(type_params=self.type_paramsA, name=c_name, run_type='vaccination',
                                  beta=c_beta_base, death_coefficient=c_death_base, arrival_coefficient=c_arrival_base,
                                  symptomatic_coefficient=spc, vaccine_priority=self.priority_vaccine_scenarios[pvs],
                                  vaccine_information=self.vaccine_information,
                                  vaccine_start_days=self.vaccine_start_days,
                                  vaccine_end_days=self.vaccine_end_days, sim_length=365 * 3, use_tqdm=True,
                                  t_lost_inm=0, n_parallel=multiprocessing.cpu_count(),
                                  exporting_information=[], vaccine_effectiveness_params=vac_eff_params)
        print('End process')

    def run_sensibility_ifr(self, c_beta_base: tuple, c_death_base: tuple, c_arrival_base: tuple, spc: float):
        for ifr_change in [0.8, 0.9, 1.1, 1.2]:
            death_coef = []
            for dc in c_death_base:
                death_coef.append(dc*ifr_change)
            death_coef = tuple(death_coef)
            c_name = 'no_vac_spc_' + '_ifr_change_' + str(ifr_change)
            print(c_name, datetime.datetime.now())
            self.model_ex.run(type_params=self.type_paramsA, name=c_name, run_type='no_vaccination',
                              beta=c_beta_base, death_coefficient=death_coef, arrival_coefficient=c_arrival_base,
                              symptomatic_coefficient=spc, vaccine_priority=None, vaccine_information=None,
                              vaccine_start_days=None, vaccine_end_days=None, sim_length=365 * 3, use_tqdm=True,
                              t_lost_inm=0, n_parallel=34,
                              exporting_information=[])
            for pvs in self.priority_vaccine_scenarios:
                c_name = 'vac_priority_' + pvs + '_ifr_change_' + str(ifr_change)
                print(c_name, datetime.datetime.now())
                self.model_ex.run(type_params=self.type_paramsA, name=c_name, run_type='vaccination',
                                  beta=c_beta_base, death_coefficient=death_coef, arrival_coefficient=c_arrival_base,
                                  symptomatic_coefficient=spc, vaccine_priority=self.priority_vaccine_scenarios[pvs],
                                  vaccine_information=self.vaccine_information,
                                  vaccine_start_days=self.vaccine_start_days,
                                  vaccine_end_days=self.vaccine_end_days, sim_length=365 * 3, use_tqdm=True,
                                  t_lost_inm=0, n_parallel=multiprocessing.cpu_count(),
                                  exporting_information=[])
        print('End process')

    def run_sensibility_vac_end(self, c_beta_base: tuple, c_death_base: tuple, c_arrival_base: tuple, spc: float):
        dates = {'A2021M8': 541, 'A2021M9': 572, 'A2021M10': 602, 'A2021M11': 633, 'A2021M12': 663, 'A2022M1': 694,
                 'A2022M3': 753, 'A2022M4': 784, 'A2022M5': 814, 'A2022M6': 845, 'A2022M7': 875}
        for vac_end in dates:
            for pvs in self.priority_vaccine_scenarios:
                c_name = 'vac_priority_' + pvs + '_vac_end_day_' + vac_end
                print(c_name, datetime.datetime.now())
                self.model_ex.run(type_params=self.type_paramsA, name=c_name, run_type='vaccination',
                                  beta=c_beta_base, death_coefficient=c_death_base, arrival_coefficient=c_arrival_base,
                                  symptomatic_coefficient=spc, vaccine_priority=self.priority_vaccine_scenarios[pvs],
                                  vaccine_information=self.vaccine_information,
                                  vaccine_start_days=self.vaccine_start_days,
                                  vaccine_end_days={'BASE_VALUE': dates[vac_end]}, sim_length=365 * 3, use_tqdm=True,
                                  t_lost_inm=0, n_parallel=multiprocessing.cpu_count(),
                                  exporting_information=[])
        print('End process')

    def run_sensibility_contact_variation(self, c_beta_base: tuple, c_death_base: tuple, c_arrival_base: tuple,
                                          spc: float):
        deltas = {'0': 0, '-10': -0.1, '-25': -0.25, '-50': -0.5, 'last_date': None}
        for delta in deltas:
            c_name = 'no_vac_CV_change_' + delta
            print(c_name, datetime.datetime.now())
            self.model_ex.run(type_params=self.type_paramsA, name=c_name, run_type='no_vaccination',
                              beta=c_beta_base, death_coefficient=c_death_base, arrival_coefficient=c_arrival_base,
                              symptomatic_coefficient=spc, vaccine_priority=None, vaccine_information=None,
                              vaccine_start_days=None, vaccine_end_days=None, sim_length=365 * 3, use_tqdm=True,
                              t_lost_inm=0, n_parallel=34,
                              exporting_information=[], future_variation=deltas[delta])
            for pvs in self.priority_vaccine_scenarios:
                c_name = 'vac_priority_' + pvs + '_CV_change_' + delta
                print(c_name, datetime.datetime.now())
                self.model_ex.run(type_params=self.type_paramsA, name=c_name, run_type='vaccination',
                                  beta=c_beta_base, death_coefficient=c_death_base, arrival_coefficient=c_arrival_base,
                                  symptomatic_coefficient=spc, vaccine_priority=self.priority_vaccine_scenarios[pvs],
                                  vaccine_information=self.vaccine_information,
                                  vaccine_start_days=self.vaccine_start_days,
                                  vaccine_end_days=self.vaccine_end_days, sim_length=365 * 3, use_tqdm=True,
                                  t_lost_inm=0, n_parallel=multiprocessing.cpu_count(),
                                  exporting_information=[], future_variation=deltas[delta])
        print('End process')
