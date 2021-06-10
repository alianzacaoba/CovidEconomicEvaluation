import multiprocessing
import os.path

from logic.model import Model
import pandas as pd
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
        type_params_list = ['attention_cost', 'vaccine_cost', 'daly', 'initial_sus', ('p_c_d', 'S'), ('p_c_d', 'H'),
                            ('p_h_d', 'S'), ('p_h_d', 'H'), ('p_i_d', 'S'), ('p_i_d', 'H'), ('p_s', 'e0'),
                            ('p_s', 'e1'), ('p_s', 'e2'), ('p_s', 'e3'), ('p_s', 'e4'), ('p_s', 'e5'), ('p_s', 'e6'),
                            ('p_s', 'e7'), 'p_i', 'p_h', 't_a', 't_sy', 'vaccine_day', 'vac_death',
                            't_e', 't_p', 't_ri', 't_d', 't_r']
        for pv in type_params_list:
            self.type_paramsA[pv] = 'BASE_VALUE'

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
        self.vaccine_start_days = {'BASE_VALUE': 419}
        self.vaccine_end_days = {'INF_VALUE': 690, 'BASE_VALUE': 780, 'MAX_VALUE': 870}
        self.n_vaccine_days = 305
        self.type_paramsA['daly'] = 'BASE_VALUE'
        self.type_paramsA['cost'] = 'BASE_VALUE'
        self.type_paramsA['vaccine_day'] = 'BASE_VALUE'
        vaccine_information = pd.read_excel(DIR_INPUT + 'vaccine_info.xlsx')
        self.vaccine_information = dict()
        for v_scenario in vaccine_information['vaccine_info_scenario'].unique():
            df = vaccine_information[vaccine_information['vaccine_info_scenario'] == v_scenario].copy()
            df.drop(columns='vaccine_info_scenario', inplace=True)
            df.set_index('vaccine', inplace=True)
            self.vaccine_information[v_scenario] = df.to_dict('index')

    def run(self, c_beta_base: tuple, c_death_base: tuple, c_arrival_base: tuple, spc: float):
        """
        Simulation with the base cases for every scenario.
        :param c_beta_base: Beta values for each of the regions
        :param c_death_base: Death coefficients values for each of the regions
        :param c_arrival_base: Arrival increment coefficients for each of the regions
        :param spc: Symptomatic increase/decrease coefficient
        """
        c_name = 'result_no_vac'
        print(c_name, datetime.datetime.now())
        self.model_ex.run(type_params=self.type_paramsA, name=c_name, run_type='no_vaccination',
                              beta=c_beta_base, death_coefficient=c_death_base, arrival_coefficient=c_arrival_base,
                              symptomatic_coefficient=spc, vaccine_priority=None, vaccine_information=None,
                              vaccine_start_days=None, vaccine_end_days=None, sim_length=365 * 3, use_tqdm=True,
                              t_lost_imm=0, n_parallel=multiprocessing.cpu_count(),
                              exporting_information='All')
        for pvs in self.priority_vaccine_scenarios:
            c_name = 'result_vac_priority_' + pvs
            print(c_name, datetime.datetime.now())
            self.model_ex.run(type_params=self.type_paramsA, name=c_name, run_type='vaccination',
                              beta=c_beta_base, death_coefficient=c_death_base, arrival_coefficient=c_arrival_base,
                              symptomatic_coefficient=spc, vaccine_priority=self.priority_vaccine_scenarios[pvs],
                              vaccine_information=self.vaccine_information['base'],
                              vaccine_start_days=self.vaccine_start_days,
                              vaccine_end_days=self.vaccine_end_days, sim_length=365 * 3, use_tqdm=True,
                              t_lost_imm=0, n_parallel=multiprocessing.cpu_count(),
                              exporting_information='All')
        print('End process')

    def run_immunity_loss_scenarios(self, c_beta_base: tuple, c_death_base: tuple, c_arrival_base: tuple, spc: float):
        """
        Simulation for the immunity loss sensitivity scenarios
        :param c_beta_base: Beta values for each of the regions
        :param c_death_base: Death coefficients values for each of the regions
        :param c_arrival_base: Arrival increment coefficients for each of the regions
        :param spc: Symptomatic increase/decrease coefficient
        """

        if not os.path.exists(os.path.join(DIR_OUTPUT, 'immunity_loss')):
            os.makedirs(os.path.join(DIR_OUTPUT, 'immunity_loss'))
        for i in range(0, 11):
            c_name = 'result_no_vac_' + str(i * 5)
            print(c_name, datetime.datetime.now())
            self.model_ex.run(type_params=self.type_paramsA, name=os.path.join('immunity_loss', c_name),
                              run_type='no_vaccination', beta=c_beta_base, death_coefficient=c_death_base,
                              arrival_coefficient=c_arrival_base, symptomatic_coefficient=spc, vaccine_priority=None,
                              vaccine_information=None, vaccine_start_days=None, vaccine_end_days=None,
                              sim_length=365 * 3, use_tqdm=True, t_lost_imm=i * 5 / (100 * 365),
                              n_parallel=multiprocessing.cpu_count(), exporting_information=[])
            for pvs in self.priority_vaccine_scenarios:
                c_name = 'result_vac_priority_' + pvs + '_' + str(i * 5)
                print(c_name, datetime.datetime.now())
                self.model_ex.run(type_params=self.type_paramsA, name=os.path.join('immunity_loss', c_name),
                                  run_type='vaccination', beta=c_beta_base, death_coefficient=c_death_base,
                                  arrival_coefficient=c_arrival_base, symptomatic_coefficient=spc,
                                  vaccine_priority=self.priority_vaccine_scenarios[pvs],
                                  vaccine_information=self.vaccine_information['base'],
                                  vaccine_start_days=self.vaccine_start_days,
                                  vaccine_end_days=self.vaccine_end_days, sim_length=365 * 3, use_tqdm=True,
                                  t_lost_imm=i * 5 / (100 * 365), n_parallel=multiprocessing.cpu_count(),
                                  exporting_information=[])
        print('End process')

    def run_tornado(self, c_beta_base: tuple, c_death_base: tuple, c_arrival_base: tuple, spc: float):
        """
        Simulation for the tornado sensitivity analysis for every scenario.
        :param c_beta_base: Beta values for each of the regions
        :param c_death_base: Death coefficients values for each of the regions
        :param c_arrival_base: Arrival increment coefficients for each of the regions
        :param spc: Symptomatic increase/decrease coefficient
        """
        if not os.path.exists(os.path.join(DIR_OUTPUT, 'tornado')):
            os.makedirs(os.path.join(DIR_OUTPUT, 'tornado'))
        c_name = 'result_no_vac_base_case'
        print(c_name, datetime.datetime.now())
        self.model_ex.run(type_params=self.type_paramsA, name=os.path.join('tornado', c_name),
                          run_type='no_vaccination', beta=c_beta_base, death_coefficient=c_death_base,
                          arrival_coefficient=c_arrival_base, symptomatic_coefficient=spc, vaccine_priority=None,
                          vaccine_information=None, vaccine_start_days=None, vaccine_end_days=None, sim_length=365 * 3,
                          use_tqdm=True, t_lost_imm=0, n_parallel=multiprocessing.cpu_count(), exporting_information=[])
        for pvs in self.priority_vaccine_scenarios:
            c_name = 'result_vac_priority_' + pvs + '_base_case'
            print(c_name, datetime.datetime.now())
            self.model_ex.run(type_params=self.type_paramsA, name=os.path.join('tornado', c_name),
                              run_type='vaccination', beta=c_beta_base, death_coefficient=c_death_base,
                              arrival_coefficient=c_arrival_base, symptomatic_coefficient=spc,
                              vaccine_priority=self.priority_vaccine_scenarios[pvs],
                              vaccine_information=self.vaccine_information['base'],
                              vaccine_start_days=self.vaccine_start_days, vaccine_end_days=self.vaccine_end_days,
                              sim_length=365 * 3, use_tqdm=True, t_lost_imm=0, n_parallel=multiprocessing.cpu_count(),
                              exporting_information=[])
        for pv in ['t_e', 't_p', 't_ri', 't_d', 't_r']:  # self.type_paramsA:
            type_params_b = self.type_paramsA.copy()
            for val in ['INF_VALUE', 'MAX_VALUE']:
                type_params_b[pv] = val
                if type(pv) is tuple:
                    c_name = 'result_sensitivity_'
                    for c in pv:
                        c_name += c + '_'
                    c_name += val + '_vac_priority_no_vac'
                else:
                    c_name = 'result_sensitivity_' + pv + '_' + val + '_vac_priority_no_vac'
                print(c_name, datetime.datetime.now())
                self.model_ex.run(type_params=type_params_b, name=os.path.join('tornado', c_name),
                                  run_type='no_vaccination', beta=c_beta_base, death_coefficient=c_death_base,
                                  arrival_coefficient=c_arrival_base, symptomatic_coefficient=spc,
                                  vaccine_priority=None, vaccine_information=None, vaccine_start_days=None,
                                  vaccine_end_days=None, sim_length=365 * 3, use_tqdm=True, t_lost_imm=0,
                                  n_parallel=multiprocessing.cpu_count(), exporting_information=[])
                for pvs in self.priority_vaccine_scenarios:
                    if type(pv) is tuple:
                        c_name = 'result_sensitivity_'
                        for c in pv:
                            c_name += c + '_'
                        c_name += val + '_vac_priority_' + pvs
                    else:
                        c_name = 'result_sensitivity_' + pv + '_' + val + '_vac_priority_' + pvs
                    print(c_name, datetime.datetime.now())
                    self.model_ex.run(type_params=type_params_b, name=os.path.join('tornado', c_name),
                                      run_type='vaccination', beta=c_beta_base, death_coefficient=c_death_base,
                                      arrival_coefficient=c_arrival_base, symptomatic_coefficient=spc,
                                      vaccine_priority=self.priority_vaccine_scenarios[pvs],
                                      vaccine_information=self.vaccine_information['base'],
                                      vaccine_start_days=self.vaccine_start_days,
                                      vaccine_end_days=self.vaccine_end_days, sim_length=365 * 3, use_tqdm=True,
                                      t_lost_imm=0, n_parallel=multiprocessing.cpu_count(),
                                      exporting_information=[])
        '''ifr_changes = {'INF_VALUE': 0.8, 'SUP_VALUE': 1.2}
        for ifr_change in ifr_changes:
            death_coef = []
            for dc in c_death_base:
                death_coef.append(dc*ifr_changes[ifr_change])
            death_coef = tuple(death_coef)
            c_name = 'result_sensitivity_ifr_change_' + ifr_change + '_vac_priority_no_vac'
            print(c_name, datetime.datetime.now())
            self.model_ex.run(type_params=self.type_paramsA, name=os.path.join('tornado', c_name),
                              run_type='no_vaccination', beta=c_beta_base, death_coefficient=death_coef,
                              arrival_coefficient=c_arrival_base, symptomatic_coefficient=spc, vaccine_priority=None,
                              vaccine_information=None, vaccine_start_days=None, vaccine_end_days=None,
                              sim_length=365 * 3, use_tqdm=True, t_lost_imm=0, n_parallel=multiprocessing.cpu_count(),
                              exporting_information=[])
            for pvs in self.priority_vaccine_scenarios:
                c_name = 'result_sensitivity_ifr_change_' + ifr_change + '_vac_priority_' + pvs
                print(c_name, datetime.datetime.now())
                self.model_ex.run(type_params=self.type_paramsA, name=os.path.join('tornado', c_name),
                                  run_type='vaccination', beta=c_beta_base, death_coefficient=death_coef,
                                  arrival_coefficient=c_arrival_base, symptomatic_coefficient=spc,
                                  vaccine_priority=self.priority_vaccine_scenarios[pvs],
                                  vaccine_information=self.vaccine_information['base'],
                                  vaccine_start_days=self.vaccine_start_days, vaccine_end_days=self.vaccine_end_days,
                                  sim_length=365 * 3, use_tqdm=True, t_lost_imm=0,
                                  n_parallel=multiprocessing.cpu_count(), exporting_information=[])
        immunity_loss = {'INF_VALUE': 1 / 1825, 'MAX_VALUE': 0}
        for im in immunity_loss:
            c_name = 'result_sensitivity_immunity_loss_' + im + '_vac_priority_no_vac'
            print(c_name, datetime.datetime.now())
            self.model_ex.run(type_params=self.type_paramsA, name=os.path.join('tornado', c_name),
                              run_type='no_vaccination', beta=c_beta_base, death_coefficient=c_death_base,
                              arrival_coefficient=c_arrival_base, symptomatic_coefficient=spc, vaccine_priority=None,
                              vaccine_information=None, vaccine_start_days=None, vaccine_end_days=None,
                              sim_length=365 * 3, use_tqdm=True, t_lost_imm=immunity_loss[im],
                              n_parallel=multiprocessing.cpu_count(), exporting_information=[])
            for pvs in self.priority_vaccine_scenarios:
                c_name = 'result_sensitivity_immunity_loss_' + im + '_vac_priority_' + pvs
                print(c_name, datetime.datetime.now())
                self.model_ex.run(type_params=self.type_paramsA, name=os.path.join('tornado', c_name),
                                  run_type='vaccination', beta=c_beta_base, death_coefficient=c_death_base,
                                  arrival_coefficient=c_arrival_base, symptomatic_coefficient=spc,
                                  vaccine_priority=self.priority_vaccine_scenarios[pvs],
                                  vaccine_information=self.vaccine_information['base'],
                                  vaccine_start_days=self.vaccine_start_days, vaccine_end_days=self.vaccine_end_days,
                                  sim_length=365 * 3, use_tqdm=True, t_lost_imm=immunity_loss[im],
                                  n_parallel=multiprocessing.cpu_count(), exporting_information=[])
        vac_effs = {'INF_VALUE': ['R', 0.2], 'SUP_VALUE': ['R', -0.2]}
        for vac_eff_params in vac_effs:
            for pvs in self.priority_vaccine_scenarios:
                c_name = 'result_sensitivity_vaccineEffectiveness_' + vac_eff_params + '_vac_priority_' + pvs
                print(c_name, datetime.datetime.now())
                self.model_ex.run(type_params=self.type_paramsA, name=os.path.join('tornado', c_name),
                                  run_type='vaccination', beta=c_beta_base, death_coefficient=c_death_base,
                                  arrival_coefficient=c_arrival_base, symptomatic_coefficient=spc,
                                  vaccine_priority=self.priority_vaccine_scenarios[pvs],
                                  vaccine_information=self.vaccine_information['base'],
                                  vaccine_start_days=self.vaccine_start_days, vaccine_end_days=self.vaccine_end_days,
                                  sim_length=365 * 3, use_tqdm=True, t_lost_imm=0,
                                  n_parallel=multiprocessing.cpu_count(), exporting_information=[],
                                  vaccine_effectiveness_params=vac_effs[vac_eff_params])'''
        print('End process')

    def run_quality_test(self, c_beta_base: tuple, c_death_base: tuple, c_arrival_base: tuple, spc: float,
                         name: str = 'no_vac'):
        self.model_ex.run(type_params=self.type_paramsA, name=name, run_type='no_vaccination', beta=c_beta_base,
                          death_coefficient=c_death_base, arrival_coefficient=c_arrival_base,
                          symptomatic_coefficient=spc, sim_length=500, n_parallel=multiprocessing.cpu_count(),
                          use_tqdm=True)
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
        if not os.path.exists(os.path.join(DIR_OUTPUT, 'symptomatic_coefficient_change')):
            os.makedirs(os.path.join(DIR_OUTPUT, 'symptomatic_coefficient_change'))
        for cv in [0.5, 0.75, 1.5, 1.75]:
            c_name = 'result_no_vac_spc_' + str(cv)
            print(c_name, datetime.datetime.now())
            self.model_ex.run(type_params=self.type_paramsA, name=os.path.join('symptomatic_coefficient_change',
                                                                               c_name), run_type='no_vaccination',
                              beta=c_beta_base, death_coefficient=c_death_base, arrival_coefficient=c_arrival_base,
                              symptomatic_coefficient=spc*cv, vaccine_priority=None, vaccine_information=None,
                              vaccine_start_days=None, vaccine_end_days=None, sim_length=365 * 3, use_tqdm=True,
                              t_lost_imm=0, n_parallel=multiprocessing.cpu_count(), exporting_information=[])
            for pvs in self.priority_vaccine_scenarios:
                c_name = 'result_vac_priority_' + pvs + '_spc_' + str(cv)
                print(c_name, datetime.datetime.now())
                self.model_ex.run(type_params=self.type_paramsA, name=os.path.join('symptomatic_coefficient_change',
                                                                                   c_name), run_type='vaccination',
                                  beta=c_beta_base, death_coefficient=c_death_base, arrival_coefficient=c_arrival_base,
                                  symptomatic_coefficient=spc*cv, 
                                  vaccine_priority=self.priority_vaccine_scenarios[pvs], 
                                  vaccine_information=self.vaccine_information['base'],
                                  vaccine_start_days=self.vaccine_start_days,
                                  vaccine_end_days=self.vaccine_end_days, sim_length=365 * 3, use_tqdm=True,
                                  t_lost_imm=0, n_parallel=multiprocessing.cpu_count(),
                                  exporting_information=[])
        print('End process')

    def run_sensibility_vac_eff(self, c_beta_base: tuple, c_death_base: tuple, c_arrival_base: tuple, spc: float):
        if not os.path.exists(os.path.join(DIR_OUTPUT, 'vaccine_effectiveness')):
            os.makedirs(os.path.join(DIR_OUTPUT, 'vaccine_effectiveness'))
        for vac_eff_params in [['R', 0.25], ['R', 0.50], ['V', 0.5], ['V', 0.7], ['V', 0.9]]:
            for pvs in self.priority_vaccine_scenarios:
                c_name = 'result_vac_priority_' + pvs + '_veff_' + vac_eff_params[0] + '_' + str(vac_eff_params[1])
                print(c_name, datetime.datetime.now())
                self.model_ex.run(type_params=self.type_paramsA, name=os.path.join('vaccine_effectiveness', c_name),
                                  run_type='vaccination', beta=c_beta_base, death_coefficient=c_death_base,
                                  arrival_coefficient=c_arrival_base, symptomatic_coefficient=spc,
                                  vaccine_priority=self.priority_vaccine_scenarios[pvs],
                                  vaccine_information=self.vaccine_information['base'],
                                  vaccine_start_days=self.vaccine_start_days,
                                  vaccine_end_days=self.vaccine_end_days, sim_length=365 * 3, use_tqdm=True,
                                  t_lost_imm=0, n_parallel=multiprocessing.cpu_count(),
                                  exporting_information=[], vaccine_effectiveness_params=vac_eff_params)
        print('End process')

    def run_sensibility_ifr(self, c_beta_base: tuple, c_death_base: tuple, c_arrival_base: tuple, spc: float):
        if not os.path.exists(os.path.join(DIR_OUTPUT, 'IFR_Change')):
            os.makedirs(os.path.join(DIR_OUTPUT, 'IFR_Change'))
        for ifr_change in [0.8, 0.9, 1.1, 1.2]:
            death_coef = []
            for dc in c_death_base:
                death_coef.append(dc*ifr_change)
            death_coef = tuple(death_coef)
            c_name = 'result_no_vac_spc_' + '_ifr_change_' + str(ifr_change)
            print(c_name, datetime.datetime.now())
            self.model_ex.run(type_params=self.type_paramsA, name=os.path.join('IFR_Change', c_name),
                              run_type='no_vaccination', beta=c_beta_base, death_coefficient=death_coef,
                              arrival_coefficient=c_arrival_base, symptomatic_coefficient=spc, vaccine_priority=None,
                              vaccine_information=None, vaccine_start_days=None, vaccine_end_days=None,
                              sim_length=365 * 3, use_tqdm=True, t_lost_imm=0, n_parallel=multiprocessing.cpu_count(),
                              exporting_information=[])
            for pvs in self.priority_vaccine_scenarios:
                c_name = 'result_vac_priority_' + pvs + '_ifr_change_' + str(ifr_change)
                print(c_name, datetime.datetime.now())
                self.model_ex.run(type_params=self.type_paramsA, name=os.path.join('IFR_Change', c_name),
                                  run_type='vaccination', beta=c_beta_base, death_coefficient=death_coef,
                                  arrival_coefficient=c_arrival_base, symptomatic_coefficient=spc,
                                  vaccine_priority=self.priority_vaccine_scenarios[pvs],
                                  vaccine_information=self.vaccine_information['base'],
                                  vaccine_start_days=self.vaccine_start_days,
                                  vaccine_end_days=self.vaccine_end_days, sim_length=365 * 3, use_tqdm=True,
                                  t_lost_imm=0, n_parallel=multiprocessing.cpu_count(),
                                  exporting_information=[])
        print('End process')

    def run_sensibility_vac_end(self, c_beta_base: tuple, c_death_base: tuple, c_arrival_base: tuple, spc: float):
        if not os.path.exists(os.path.join(DIR_OUTPUT, 'vaccine_end_day')):
            os.makedirs(os.path.join(DIR_OUTPUT, 'vaccine_end_day'))
        dates = {'A2021M10': 602, 'A2022M1': 694, 'A2022M4': 784, 'A2022M7': 875}
        for vac_end in dates:
            for pvs in self.priority_vaccine_scenarios:
                c_name = 'result_vac_priority_' + pvs + '_vac_end_day_' + vac_end
                print(c_name, datetime.datetime.now())
                self.model_ex.run(type_params=self.type_paramsA, name=os.path.join('vaccine_end_day', c_name),
                                  run_type='vaccination', beta=c_beta_base, death_coefficient=c_death_base,
                                  arrival_coefficient=c_arrival_base, symptomatic_coefficient=spc,
                                  vaccine_priority=self.priority_vaccine_scenarios[pvs],
                                  vaccine_information=self.vaccine_information['base'],
                                  vaccine_start_days=self.vaccine_start_days,
                                  vaccine_end_days={'BASE_VALUE': dates[vac_end]}, sim_length=365 * 3, use_tqdm=True,
                                  t_lost_imm=0, n_parallel=multiprocessing.cpu_count(),
                                  exporting_information=[])
        print('End process')

    def run_sensibility_contact_variation(self, c_beta_base: tuple, c_death_base: tuple, c_arrival_base: tuple,
                                          spc: float):
        if not os.path.exists(os.path.join(DIR_OUTPUT, 'contact_coefficient_change')):
            os.makedirs(os.path.join(DIR_OUTPUT, 'contact_coefficient_change'))
        deltas = {'0': 0, '-10': -0.1, '-25': -0.25, '-50': -0.5, 'last_date': None}
        for delta in deltas:
            c_name = 'result_no_vac_CV_change_' + delta
            print(c_name, datetime.datetime.now())
            self.model_ex.run(type_params=self.type_paramsA, name=os.path.join('contact_coefficient_change', c_name),
                              run_type='no_vaccination', beta=c_beta_base, death_coefficient=c_death_base,
                              arrival_coefficient=c_arrival_base, symptomatic_coefficient=spc, vaccine_priority=None,
                              vaccine_information=None, vaccine_start_days=None, vaccine_end_days=None,
                              sim_length=365 * 3, use_tqdm=True, t_lost_imm=0, n_parallel=multiprocessing.cpu_count(),
                              exporting_information=[], future_variation=deltas[delta])
            for pvs in self.priority_vaccine_scenarios:
                c_name = 'result_vac_priority_' + pvs + '_CV_change_' + delta
                print(c_name, datetime.datetime.now())
                self.model_ex.run(type_params=self.type_paramsA, name=os.path.join('contact_coefficient_change',
                                                                                   c_name), run_type='vaccination',
                                  beta=c_beta_base, death_coefficient=c_death_base, arrival_coefficient=c_arrival_base,
                                  symptomatic_coefficient=spc, vaccine_priority=self.priority_vaccine_scenarios[pvs],
                                  vaccine_information=self.vaccine_information['base'],
                                  vaccine_start_days=self.vaccine_start_days,
                                  vaccine_end_days=self.vaccine_end_days, sim_length=365 * 3, use_tqdm=True,
                                  t_lost_imm=0, n_parallel=multiprocessing.cpu_count(),
                                  exporting_information=[], future_variation=deltas[delta])
        print('End process')

    def run_sensibility_contact_and_immunity_variation(self, c_beta_base: tuple, c_death_base: tuple,
                                                       c_arrival_base: tuple, spc: float):
        if not os.path.exists(os.path.join(DIR_OUTPUT, 'contact_matrix_immunity_loss')):
            os.makedirs(os.path.join(DIR_OUTPUT, 'contact_matrix_immunity_loss'))
        contact_variations = {'-25': -0.25, 'last_date': None}
        immunity_loss = {'10': 1/3650, '25': 1/1460}
        for cv in contact_variations:
            for im in immunity_loss:
                c_name = 'result_no_vac_CV_' + cv + '_immunity_loss_' + im
                print(c_name, datetime.datetime.now())
                self.model_ex.run(type_params=self.type_paramsA, name=os.path.join('contact_matrix_immunity_loss',
                                                                                   c_name), run_type='no_vaccination',
                                  beta=c_beta_base, death_coefficient=c_death_base, arrival_coefficient=c_arrival_base,
                                  symptomatic_coefficient=spc, vaccine_priority=None, vaccine_information=None,
                                  vaccine_start_days=None, vaccine_end_days=None, sim_length=365 * 3, use_tqdm=True,
                                  t_lost_imm=immunity_loss[im], n_parallel=multiprocessing.cpu_count(),
                                  exporting_information=[], future_variation=contact_variations[cv])
                for pvs in self.priority_vaccine_scenarios:
                    c_name = 'result_vac_priority_' + pvs + '_CV_' + cv + '_immunity_loss_' + im
                    print(c_name, datetime.datetime.now())
                    self.model_ex.run(type_params=self.type_paramsA, name=os.path.join('contact_matrix_immunity_loss',
                                                                                   c_name), run_type='vaccination',
                                      beta=c_beta_base, death_coefficient=c_death_base,
                                      arrival_coefficient=c_arrival_base, symptomatic_coefficient=spc,
                                      vaccine_priority=self.priority_vaccine_scenarios[pvs],
                                      vaccine_information=self.vaccine_information['base'],
                                      vaccine_start_days=self.vaccine_start_days,
                                      vaccine_end_days=self.vaccine_end_days, sim_length=365 * 3, use_tqdm=True,
                                      t_lost_imm=immunity_loss[im], n_parallel=multiprocessing.cpu_count(),
                                      exporting_information=[], future_variation=contact_variations[cv])
        print('End process')

    def run_sensibility_ifr_and_vac_end_variation(self, c_beta_base: tuple, c_death_base: tuple,
                                                       c_arrival_base: tuple, spc: float):
        if not os.path.exists(os.path.join(DIR_OUTPUT, 'vaccine_end_ifr_change')):
            os.makedirs(os.path.join(DIR_OUTPUT, 'vaccine_end_ifr_change'))
        dates = {'A2021M10': 602, 'A2022M1': 694, 'A2022M4': 784, 'A2022M7': 875}
        ifr_changes = {'0.9': 0.9, '1.1': 1.1}
        for ifr in ifr_changes:
            death_coef = []
            for dc in c_death_base:
                death_coef.append(dc * ifr_changes[ifr])
            death_coef = tuple(death_coef)
            for dv in dates:
                for pvs in self.priority_vaccine_scenarios:
                    c_name = 'result_vac_priority_' + pvs + '_vac_end_' + dv + '_ifr_change_' + ifr
                    print(c_name, datetime.datetime.now())
                    self.model_ex.run(type_params=self.type_paramsA, name=os.path.join('vaccine_end_ifr_change',
                                                                                       c_name), run_type='vaccination',
                                      beta=c_beta_base, death_coefficient=death_coef,
                                      arrival_coefficient=c_arrival_base, symptomatic_coefficient=spc,
                                      vaccine_priority=self.priority_vaccine_scenarios[pvs],
                                      vaccine_information=self.vaccine_information['base'],
                                      vaccine_start_days=self.vaccine_start_days,
                                      vaccine_end_days={'BASE_VALUE': dates[dv]}, sim_length=365 * 3, use_tqdm=True,
                                      t_lost_imm=0, n_parallel=multiprocessing.cpu_count(), exporting_information=[],
                                      future_variation=None)
        print('End process')

    def run_sensibility_vaccine_dist(self, c_beta_base: tuple, c_death_base: tuple, c_arrival_base: tuple, spc: float):
        if not os.path.exists(os.path.join(DIR_OUTPUT, 'vaccine_distributions')):
            os.makedirs(os.path.join(DIR_OUTPUT, 'vaccine_distributions'))
        dates = {'A2021M10': 602, 'A2022M1': 694, 'A2022M4': 784, 'A2022M7': 875}
        contact_variations = {'-50': -0.5, 'last_date': None}
        for v in self.vaccine_information.keys():
            for vac_end in dates:
                for cv in contact_variations:
                    for pvs in self.priority_vaccine_scenarios:
                        c_name = 'result_vac_priority_' + pvs + '_vac_dist_' + v + '_vac_end_day_' + vac_end + '_cv_' + cv
                        print(c_name, datetime.datetime.now())
                        self.model_ex.run(type_params=self.type_paramsA, name=os.path.join('vaccine_distributions',
                                                                                           c_name),
                                          run_type='vaccination', beta=c_beta_base, death_coefficient=c_death_base,
                                          arrival_coefficient=c_arrival_base, symptomatic_coefficient=spc,
                                          vaccine_priority=self.priority_vaccine_scenarios[pvs],
                                          vaccine_information=self.vaccine_information[v],
                                          vaccine_start_days=self.vaccine_start_days,
                                          vaccine_end_days={'BASE_VALUE': dates[vac_end]}, sim_length=365 * 3,
                                          use_tqdm=True, t_lost_imm=0, n_parallel=multiprocessing.cpu_count(),
                                          exporting_information=[], future_variation=contact_variations[cv])
        print('End process')
