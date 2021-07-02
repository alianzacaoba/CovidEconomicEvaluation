import datetime
import math
import multiprocessing
from typing import List, Any, Union
from logic.compartment import Compartment
import pandas as pd
import numpy as np
import threading
from threading import Thread, Lock
from tqdm import tqdm
import warnings
import time
from root import DIR_INPUT, DIR_OUTPUT

warnings.simplefilter('error')


beta_f = (0.022007606119173795, 0.017863980243733434, 0.018196130566806898, 0.018457626077325776, 0.01748616435199459,
          0.016227115077950355)
dc_f = (1.1323965306925503, 0.7125429486836731, 1.0444436705577909, 1.0455277531926157, 0.8937395613665182,
        0.9248158502415792)
arrival_f = (15.610984192361858, 7.118033263153407, 13.580052334837838, 6.872622856121195, 19.179202373513895,
             23.821317070305813)
spc_f = 0.2761426379037166
# error_cases : (0.01144914782225527, 0.04633153074628225, 0.035662564790543826, 0.01663383341316379,
# 0.044033713992227476, 0.05507748681553992)
# error_deaths : (0.027622229848766704, 0.009587082330408638, 0.012646483805184615, 0.01921871618354314,
# 0.062333297880331026, 0.07248222923337995)
# error : 0.03471753964393546


def calculate_vaccine_assignments(population: dict, day: int, vaccine_priority: list, vaccine_capacity: dict,
                                  candidates_indexes: list):
    """
    Calculates the daily vaccination assignments for every group.
    :param population: Resulting population in the simulation.
    :param day: Day to obtain information
    :param vaccine_priority: list of priorities. Each group contains a list of tuples indicating the groups that
    comprehend the priorities.
    :param vaccine_capacity: dictionary with amount for each type of vaccine and target group.
    :param candidates_indexes: indexes for states (infection related) that are candidates to the vaccine.
    :return:
    """
    day_population = dict()
    assignation = dict()
    age_groups = list(population.keys())
    work_groups = list(population[age_groups[0]].keys())
    health_groups = list(population[age_groups[0]][work_groups[0]].keys())
    vaccine_groups = list(population[age_groups[0]][work_groups[0]][health_groups[0]].keys())
    for ev in age_groups:
        for wv in work_groups:
            for hv in health_groups:
                for vv in vaccine_groups:
                    for cv in candidates_indexes:
                        day_population[(ev, wv, hv, vv, cv)] = population[ev][wv][hv][vv][cv].values[day]
    for vaccine in vaccine_capacity:
        # Diccionario con el siguiente orden: {'cual':[a_quien, cantidad]}
        current_vac = vaccine_capacity[vaccine]
        vv = current_vac[0]
        total_candidates = 0
        for ev in age_groups:
            for wv in work_groups:
                for hv in health_groups:
                    for cv in candidates_indexes:
                        total_candidates += day_population[(ev, wv, hv, vv, cv)]
        vaccines_to_use = min(total_candidates, current_vac[1])
        # Van en el orden de las listas
        for groups in vaccine_priority:
            # group = [[ev, wv, hv]]
            group_candidates = 0
            for group in groups:
                ev, wv, hv, p = group
                group_candidates += sum(day_population[(ev, wv, hv, vv, cv)] for cv in candidates_indexes)
            vaccines_for_group = min(vaccines_to_use, group_candidates)
            vaccines_to_use -= vaccines_for_group
            if group_candidates > 0:
                for group in groups:
                    ev, wv, hv, p = group
                    for cv in candidates_indexes:
                        assignation[(ev, wv, hv, vv, cv, vaccine)] = min(vaccines_for_group *
                                                                         day_population[(ev, wv, hv, vv, cv)] /
                                                                         group_candidates,
                                                                         day_population[(ev, wv, hv, vv, cv)])
                        day_population[(ev, wv, hv, vv, cv)] -= assignation[(ev, wv, hv, vv, cv, vaccine)]
    return assignation


def create_vaccine_assignation(start_date: int, end_date: int, vaccines_information: dict, population: dict,
                               vaccine_priority: list):
    """
    Creates the daily vaccine assignation considering a time period and the required information to determine dosage.
    :param population: Population information to determine the distribution
    :param start_date: Starting date of the vaccination program.
    :param end_date: End date of the vaccination program (time to reach the limit).
    :param vaccines_information: Considers the information of each vaccine ordered by name: 1) total_doses considered,
    2) period between first and second dose.
    :param vaccine_priority: Priority information to calculate the vaccine candidate population.
    :return: dictionary with the vaccine time assignation.
    """
    total_candidates = dict()
    result_information = dict()
    for gv in population.keys():
        total_candidates[gv] = 0
        for groups in vaccine_priority:
            for group in groups:
                ev, wv, hv, p = group
                total_candidates[gv] += population[gv][(ev, wv, hv)]*p
        result_information[gv] = dict()
        for day in range(start_date, end_date + 1):
            result_information[gv][day] = dict()
    total_n = sum(total_candidates[gv] for gv in population.keys())
    for gv in population.keys():
        total_candidates[gv] = total_candidates[gv]/total_n
    time_period = end_date - start_date + 1
    for v in vaccines_information.keys():
        # v is the name of the vaccine
        vac_info = vaccines_information[v]
        total_number = vac_info['total_doses']
        inter_dose_time = vac_info['inter_dose_time']
        if inter_dose_time == 0:
            daily_number = total_number/time_period
            for gv in population.keys():
                for day in range(start_date, end_date+1):
                    result_information[gv][day][v] = ['V0', daily_number*total_candidates[gv]]
        else:
            total_individuals = total_number / 2
            daily_number = total_individuals/(time_period-inter_dose_time)
            for gv in population.keys():
                for day in range(start_date, end_date + 1 - inter_dose_time):
                    result_information[gv][day][v+str(1)] = ['V0', daily_number*total_candidates[gv]]
                    result_information[gv][day+inter_dose_time][v + str(2)] = [v+str(1),
                                                                               daily_number*total_candidates[gv]]
    return result_information


class Model(object):
    age_groups: List[Any]
    departments: List[Any]
    work_groups: List[Any]

    def __init__(self):
        self.compartments = dict()
        initial_pop = pd.read_csv(DIR_INPUT + 'initial_population.csv', sep=';')
        self.departments = list(initial_pop.DEPARTMENT.unique())
        self.age_groups = list(initial_pop.AGE_GROUP.unique())
        self.work_groups = list(initial_pop.WORK_GROUP.unique())
        self.health_groups = list(initial_pop.HEALTH_GROUP.unique())
        self.initial_population = dict()
        self.regions = pd.read_csv(DIR_INPUT + 'department_regions.csv', sep=';', index_col=0).to_dict(orient='index')
        for gv in self.departments:
            self.regions[gv] = self.regions[gv]['REGION']
            self.initial_population[gv] = dict()
            for ev in self.age_groups:
                for wv in self.work_groups:
                    for hv in self.health_groups:
                        self.initial_population[gv][(ev, wv, hv)] = \
                            float(initial_pop[(initial_pop['DEPARTMENT'] == gv) & (initial_pop['AGE_GROUP'] == ev)
                                              & (initial_pop['WORK_GROUP'] == wv) & (initial_pop['HEALTH_GROUP'] == hv)
                                              ].POPULATION.sum())
        self.daly_vector = {'Home': {'INF_VALUE': 0.002, 'BASE_VALUE': 0.006, 'MAX_VALUE': 0.012},
                            'Hosp': {'INF_VALUE': 0.032, 'BASE_VALUE': 0.051, 'MAX_VALUE': 0.074},
                            'ICU': {'INF_VALUE': 0.088, 'BASE_VALUE': 0.133, 'MAX_VALUE': 0.190},
                            'InRecovery': {'INF_VALUE': 0, 'BASE_VALUE': 0, 'MAX_VALUE': 0.012},
                            'Death': {'e0': 72.11, 'e1': 62.11, 'e2': 52.11, 'e3': 42.11, 'e4': 32.11, 'e5': 22.11,
                                      'e6': 12.11, 'e7': 2.11}}
        self.vaccine_cost = {'INF_VALUE': 37167*0.8, 'BASE_VALUE': 37167, 'MAX_VALUE': 37167*1.2}
        self.contact_matrix = {'HOME': {}, 'WORK': {}, 'SCHOOL': {}, 'OTHER': {}}
        con_matrix_home = pd.read_excel(DIR_INPUT + 'contact_matrix.xlsx', sheet_name='Home', engine="openpyxl")
        con_matrix_other = pd.read_excel(DIR_INPUT + 'contact_matrix.xlsx', sheet_name='Other', engine="openpyxl")
        con_matrix_work = pd.read_excel(DIR_INPUT + 'contact_matrix.xlsx', sheet_name='Work', engine="openpyxl")
        con_matrix_school = pd.read_excel(DIR_INPUT + 'contact_matrix.xlsx', sheet_name='School', engine="openpyxl")
        self.attention_costs = pd.read_csv(DIR_INPUT+'attention_costs.csv', sep=';',
                                           index_col=[0, 1, 2]).to_dict(orient='index')

        for c in con_matrix_home.columns:
            self.contact_matrix['HOME'][c] = con_matrix_home[c].to_list()
            self.contact_matrix['OTHER'][c] = con_matrix_other[c].to_list()
            self.contact_matrix['WORK'][c] = con_matrix_work[c].to_list()
            self.contact_matrix['SCHOOL'][c] = con_matrix_school[c].to_list()
        del con_matrix_home
        del con_matrix_other
        del con_matrix_work
        del con_matrix_school
        contact_matrix_coefficients = pd.read_csv(DIR_INPUT + 'contact_matrix_coefficients.csv')
        self.max_cm_days = int(contact_matrix_coefficients.SIM_DAY.max())
        self.contact_matrix_coefficients = dict()
        for reg in contact_matrix_coefficients['REGIONS'].unique():
            cur_contact_matrix_coefficients = contact_matrix_coefficients[
                contact_matrix_coefficients['REGIONS'] == reg].copy()
            cur_contact_matrix_coefficients.drop(columns='REGIONS', inplace=True)
            cur_contact_matrix_coefficients.set_index(['SIM_DAY'], inplace=True)
            self.contact_matrix_coefficients[reg] = cur_contact_matrix_coefficients.to_dict(orient='index')

        self.birth_rates = pd.read_csv(DIR_INPUT + 'birth_rate.csv', sep=';', index_col=0).to_dict()['BIRTH_RATE']
        morbidity_frac = pd.read_csv(DIR_INPUT + 'morbidity_fraction.csv', sep=';', index_col=0)
        self.morbidity_frac = morbidity_frac.to_dict()['COMORBIDITY_RISK']
        del morbidity_frac
        death_rates = pd.read_csv(DIR_INPUT + 'death_rate.csv', sep=';')
        med_degrees = pd.read_csv(DIR_INPUT + 'medical_degrees.csv', sep=';')
        self.death_rates = dict()
        self.med_degrees = dict()
        for gv in death_rates['DEPARTMENT'].unique():
            cur_death_rate = death_rates[death_rates['DEPARTMENT'] == gv].copy()
            cur_death_rate.drop(columns='DEPARTMENT', inplace=True)
            cur_death_rate.set_index('AGE_GROUP', inplace=True)
            self.death_rates[gv] = cur_death_rate.to_dict()['DEATH_RATE']
            del cur_death_rate
            cur_med_degrees = med_degrees[med_degrees['DEPARTMENT'] == gv].copy()
            cur_med_degrees.drop(columns='DEPARTMENT', inplace=True)
            cur_med_degrees.set_index('AGE_GROUP', inplace=True)
            self.med_degrees[gv] = cur_med_degrees.to_dict(orient='index')
            del cur_med_degrees
        self.arrival_rate = pd.read_csv(DIR_INPUT + 'arrival_rate.csv', sep=';', index_col=0).to_dict()

        time_params_load = pd.read_csv(DIR_INPUT + 'input_time.csv', sep=';')
        self.time_params = dict()
        for pv1 in time_params_load['PARAMETER'].unique():
            current_param = time_params_load[time_params_load['PARAMETER'] == pv1]
            for ev in current_param.AGE_GROUP.unique():
                current_age_p = current_param[current_param['AGE_GROUP'] == ev]
                age = {'INF_VALUE': float(current_age_p['INF_VALUE'].sum()),
                       'BASE_VALUE': float(current_age_p['BASE_VALUE'].sum()),
                       'MAX_VALUE': float(current_age_p['MAX_VALUE'].sum())}
                del current_age_p
                self.time_params[(pv1, ev)] = age
                del age
            del current_param

        del time_params_load
        prob_params_load = pd.read_csv(DIR_INPUT + 'input_probabilities.csv', sep=';')
        self.prob_params = dict()
        for pv1 in prob_params_load['PARAMETER'].unique():
            current_param = prob_params_load[prob_params_load['PARAMETER'] == pv1]
            for ev in current_param.AGE_GROUP.unique():
                current_age_p = current_param[current_param['AGE_GROUP'] == ev]
                for hv in current_age_p['HEALTH_GROUP'].unique():
                    current_health = current_age_p[current_age_p['HEALTH_GROUP'] == hv]
                    values = {'INF_VALUE': float(current_health['INF_VALUE'].sum()),
                              'BASE_VALUE': float(current_health['BASE_VALUE'].sum()),
                              'MAX_VALUE': float(current_health['MAX_VALUE'].sum())}
                    self.prob_params[(pv1, ev, hv)] = values
                    del current_health
                del current_age_p
            del current_param

    def run(self, type_params: dict, name: str = 'Iteration', run_type: str = 'vaccination',
            beta: tuple = beta_f, death_coefficient: tuple = dc_f, arrival_coefficient: tuple = arrival_f,
            symptomatic_coefficient: float = spc_f, vaccine_priority: list = None, vaccine_information: dict = None,
            vaccine_start_days: dict = None, vaccine_end_days: dict = None, sim_length: int = 365 * 3,
            use_tqdm: bool = False, t_lost_imm: int = 0, n_parallel: int = multiprocessing.cpu_count()-1,
            exporting_information: Union[str, list] = 'All', vaccine_effectiveness_params: list = None,
            future_variation: float = None):

        # run_type:
        #   1) 'calibration': no vaccination status - returns the daily results of measurement (cases, deaths,
        #   seroprevalence)
        #   2) 'vaccination': model with vaccine states - requires the vaccines information characteristics.
        #   3) 'non-vaccination': model without vaccines - [0V] is the only vaccination states
        # vaccine_information debe_tener {'vacuna':{'total_doses', 'inter_dose_time', 'effectivity',
        # 'symptomatic_prob_reduction'}}
        # export_type = {'all', 'json', 'csv', 'xlsx'}
        if type(exporting_information) is str:
            if exporting_information == 'All':
                exporting_information = ['Department', 'Health', 'Work', 'Age', 'Vaccine']
            else:
                exporting_information = [exporting_information]
        vaccine_start_day = sim_length + 5
        vaccine_end_day = sim_length + 6
        if vaccine_start_days is not None and vaccine_end_days is not None:
            vaccine_start_day = vaccine_start_days['BASE_VALUE']
            vaccine_end_day = vaccine_end_days[type_params.get('vaccine_day', 'BASE_VALUE')]
        if vaccine_information is not None:
            for vac in vaccine_information:
                vaccine_eff = vaccine_information[vac]['effectivity']
                if vaccine_effectiveness_params is not None:
                    if vaccine_effectiveness_params[0] == 'R':
                        vaccine_eff *= (1-vaccine_effectiveness_params[1])
                        if vaccine_eff > 1:
                            vaccine_eff = 1
                    elif vaccine_effectiveness_params[0] == 'V':
                        vaccine_eff = vaccine_effectiveness_params[1]
                vaccine_information[vac]['dose_effectivity'] = 1 - math.sqrt(
                    1 - vaccine_eff) if vaccine_information[vac]['inter_dose_time'] > 0 else \
                    vaccine_eff

        vaccination_calendar = create_vaccine_assignation(start_date=vaccine_start_day,
                                                          end_date=vaccine_end_day,
                                                          vaccines_information=vaccine_information,
                                                          vaccine_priority=vaccine_priority,
                                                          population=self.initial_population) \
            if run_type == 'vaccination' else None
        departments = self.departments
        age_groups = self.age_groups
        work_groups = self.work_groups
        health_groups = self.health_groups
        vaccination_groups = ['V0']
        if run_type == 'vaccination':
            if 'P' in vaccine_information.keys():
                vaccination_groups.append('P1')
                vaccination_groups.append('P2')
            if 'J' in vaccine_information.keys():
                vaccination_groups.append('J')
            if 'S' in vaccine_information.keys():
                vaccination_groups.append('S1')
                vaccination_groups.append('S2')
            if 'M' in vaccine_information.keys():
                vaccination_groups.append('M1')
                vaccination_groups.append('M2')
            if 'A' in vaccine_information.keys():
                vaccination_groups.append('A1')
                vaccination_groups.append('A2')
            vaccination_groups.append('R')
        daly_vector = dict()
        for dv in self.daly_vector:
            if dv == 'Death':
                daly_vector[dv] = self.daly_vector[dv]
            else:
                daly_vector[dv] = self.daly_vector[dv][type_params.get('daly', 'BASE_VALUE')]
        t_e = self.time_params[('t_e', 'ALL')][type_params.get('t_e', 'BASE_VALUE')]
        t_p = self.time_params[('t_p', 'ALL')][type_params.get('t_p', 'BASE_VALUE')]
        t_sy = self.time_params[('t_sy', 'ALL')][type_params.get('t_sy', 'BASE_VALUE')]
        t_a = self.time_params[('t_a', 'ALL')][type_params.get('t_a', 'BASE_VALUE')]
        t_ri = self.time_params[('t_ri', 'ALL')][type_params.get('t_ri', 'BASE_VALUE')]
        initial_sus = self.prob_params[('initial_sus', 'ALL', 'ALL')][type_params.get('initial_sus', 'BASE_VALUE')]
        attention_costs = dict()
        for acv in self.attention_costs:
            attention_costs[acv] = self.attention_costs[acv][type_params.get('attention_cost', 'BASE_VALUE')]
        t_d = dict()
        t_r = dict()
        p_s = dict()
        p_c = dict()
        p_h = dict()
        p_i = dict()
        for ev in age_groups:
            t_d[ev] = self.time_params[('t_d', ev)][type_params.get('t_d', 'BASE_VALUE')]
            t_r[ev] = max(self.time_params[('t_r', ev)][type_params.get('t_r', 'BASE_VALUE')] - t_d[ev], 1)
            p_s[('V0', ev)] = min(self.prob_params[('p_s', ev, 'ALL')][type_params.get(('p_s', ev), 'BASE_VALUE')] *
                                  symptomatic_coefficient, 1.0)
            for hv in health_groups:
                p_i[(ev, 'V0', hv)] = self.prob_params[('p_i', ev, hv)][type_params.get('p_i', 'BASE_VALUE')]
                p_h[(ev, 'V0', hv)] = self.prob_params[('p_h', ev, hv)][type_params.get('p_h', 'BASE_VALUE')]
                p_c[(ev, 'V0', hv)] = 1 - p_h[(ev, 'V0', hv)] - p_i[(ev, 'V0', hv)]

        if run_type == 'vaccination':
            for ev in age_groups:
                p_s[('R', ev)] = p_s[('V0', ev)]
                for hv in health_groups:
                    p_c[(ev, 'R', hv)] = p_c[(ev, 'V0', hv)]
                    p_h[(ev, 'R', hv)] = p_h[(ev, 'V0', hv)]
                    p_i[(ev, 'R', hv)] = p_i[(ev, 'V0', hv)]
            for vv in ['P2', 'J', 'S2', 'M2', 'A2']:  # Complete process
                if vv in vaccination_groups:
                    for ev in age_groups:
                        p_s[(vv, ev)] = p_s[('V0', ev)] * vaccine_information[vv[0]]['symptomatic_prob_coef']
            for vv in ['P1', 'S1', 'M1', 'A1']:  # Half process
                if vv in vaccination_groups:
                    for ev in age_groups:
                        p_s[(vv, ev)] = (p_s[('V0', ev)] + p_s[(vv[0] + str(2), ev)]) / 2
            for ev in age_groups:
                for hv in health_groups:
                    for vv in ['P2', 'J', 'S2', 'M2', 'A2']:  # Complete process
                        if vv in vaccination_groups:
                            p_i[(ev, vv, hv)] = p_i[(ev, 'V0', hv)] * vaccine_information[vv[0]]['icu_prob_coef']
                            p_h[(ev, vv, hv)] = p_h[(ev, 'V0', hv)] * vaccine_information[vv[0]]['hosp_prob_coef']
                            p_c[(ev, vv, hv)] = 1 - p_i[(ev, vv, hv)] - p_h[(ev, vv, hv)]
                    for vv in ['P1', 'S1', 'M1', 'A1']:  # Half process
                        if vv in vaccination_groups:
                            p_i[(ev, vv, hv)] = (p_i[(ev, 'V0', hv)] + p_i[(ev, vv[0] + str(2), hv)]) / 2
                            p_h[(ev, vv, hv)] = (p_h[(ev, 'V0', hv)] + p_h[(ev, vv[0] + str(2), hv)]) / 2
                            p_c[(ev, vv, hv)] = 1 - p_i[(ev, vv, hv)] - p_h[(ev, vv, hv)]
        percentages = dict()
        if vaccine_priority is not None:
            for phase in vaccine_priority:
                for group in phase:
                    ev, wv, hv, percent = group
                    percentages[(ev, wv, hv)] = percent
        result_queue = dict()
        initial_threads = threading.active_count()
        max_threads = initial_threads + n_parallel
        workers = dict()
        results_array = np.zeros(shape=(sim_length + 1, 18)) if run_type == 'calibration' else None
        seroprevalence_array = np.zeros(shape=(sim_length + 1, 12)) if run_type == 'calibration' else None
        integrated_departments = [] if run_type == 'calibration' else None
        params = dict()
        params['run_type'] = run_type
        params['sim_length'] = sim_length
        params['groups'] = age_groups, work_groups, health_groups, vaccination_groups
        params['contact_matrix'] = self.contact_matrix
        params['t_lost_imm'] = t_lost_imm
        params['morbidity_frac'] = self.morbidity_frac
        params['exporting_information'] = exporting_information
        params['attention_costs'] = attention_costs
        params['max_cm_days'] = self.max_cm_days if future_variation is None else self.max_cm_days + 1
        if run_type == 'vaccination':
            # Vaccination run type info
            params['vaccine_information'] = vaccine_information
            params['vaccine_priority'] = vaccine_priority
            params['vaccine_cost'] = self.vaccine_cost[type_params.get('vaccine_cost', 'BASE_VALUE')]
        params['vaccination_period'] = [vaccine_start_day, vaccine_end_day]
        # All type information
        params['daly_vector'] = daly_vector
        params['time_params'] = dict()
        params['time_params']['t_e'] = t_e
        params['time_params']['t_p'] = t_p
        params['time_params']['t_sy'] = t_sy
        params['time_params']['t_a'] = t_a
        params['time_params']['t_ri'] = t_ri
        params['time_params']['t_d'] = t_d
        params['time_params']['t_r'] = t_r

        # Prob params
        params['prob_params'] = dict()
        params['prob_params']['initial_sus'] = initial_sus
        params['prob_params']['p_s'] = p_s
        params['prob_params']['p_c'] = p_c
        params['prob_params']['p_h'] = p_h
        params['prob_params']['p_i'] = p_i
        iterator = tqdm(departments) if use_tqdm else departments
        for gv in iterator:
            cur_region = self.regions[gv] - 1
            params['initial_population'] = self.initial_population[gv]
            params['contact_matrix_coefficients'] = self.contact_matrix_coefficients[self.regions[gv]].copy()
            if future_variation is not None:
                params['contact_matrix_coefficients'][self.max_cm_days+1] = dict()
                for cv in params['contact_matrix_coefficients'][self.max_cm_days]:
                    params['contact_matrix_coefficients'][self.max_cm_days + 1][cv] = future_variation
            params['beta'] = beta[cur_region]
            arrival_rate = {'START_DAY': self.arrival_rate['START_DAY'][gv],
                            'END_DAY': self.arrival_rate['START_DAY'][gv],
                            'ARRIVAL_RATE': self.arrival_rate['ARRIVAL_RATE'][gv] * arrival_coefficient[cur_region]}
            params['arrival_rate'] = arrival_rate
            params['birth_rate'] = self.birth_rates[gv]
            params['death_rate'] = self.death_rates[gv]
            params['med_degrees'] = self.med_degrees[gv]
            params['gv'] = gv
            if run_type == 'vaccination':
                # Vaccination run type info
                params['vaccination_calendar'] = vaccination_calendar[gv]
            p_c_d = dict()
            p_h_d = dict()
            p_i_d = dict()
            for ev in age_groups:
                for hv in health_groups:
                    p_c_d[(ev, 'V0', hv)] = min(death_coefficient[cur_region] * self.prob_params[('p_c_d', ev, hv)][
                        type_params.get(('p_c_d', hv), 'BASE_VALUE')], 1.0)
                    p_h_d[(ev, 'V0', hv)] = min(death_coefficient[cur_region] * self.prob_params[('p_h_d', ev, hv)][
                        type_params.get(('p_h_d', hv), 'BASE_VALUE')], 1.0)
                    p_i_d[(ev, 'V0', hv)] = min(death_coefficient[cur_region] * self.prob_params[('p_i_d', ev, hv)][
                        type_params.get(('p_i_d', hv), 'BASE_VALUE')], 1.0)
            if run_type == 'vaccination':
                vac_death_coef = 'death_coef_' + type_params.get('vac_death', 'BASE_VALUE')
                for ev in age_groups:
                    for hv in health_groups:
                        p_c_d[(ev, 'R', hv)] = p_c_d[(ev, 'V0', hv)]
                        p_h_d[(ev, 'R', hv)] = p_h_d[(ev, 'V0', hv)]
                        p_i_d[(ev, 'R', hv)] = p_i_d[(ev, 'V0', hv)]
                        for vv in ['P2', 'J', 'S2', 'M2', 'A2']:  # Half process
                            if vv in vaccination_groups:
                                p_c_d[(ev, vv, hv)] = p_c_d[(ev, 'V0', hv)] * vaccine_information[vv[0]][vac_death_coef]
                                p_h_d[(ev, vv, hv)] = p_h_d[(ev, 'V0', hv)] * vaccine_information[vv[0]][vac_death_coef]
                                p_i_d[(ev, vv, hv)] = p_i_d[(ev, 'V0', hv)] * vaccine_information[vv[0]][vac_death_coef]
                        for vv in ['P1', 'S1', 'M1', 'A1']:  # Half process
                            if vv in vaccination_groups:
                                p_c_d[(ev, vv, hv)] = (p_c_d[(ev, 'V0', hv)] + p_c_d[(ev, vv[0] + str(2), hv)]) / 2
                                p_h_d[(ev, vv, hv)] = (p_h_d[(ev, 'V0', hv)] + p_h_d[(ev, vv[0] + str(2), hv)]) / 2
                                p_i_d[(ev, vv, hv)] = (p_i_d[(ev, 'V0', hv)] + p_i_d[(ev, vv[0] + str(2), hv)]) / 2
            params['prob_params']['p_c_d'] = p_c_d
            params['prob_params']['p_h_d'] = p_h_d
            params['prob_params']['p_i_d'] = p_i_d
            if run_type == 'calibration':
                for gv2 in result_queue:
                    if gv2 not in integrated_departments:
                        gv_region = self.regions[gv2] - 1
                        results_array[:, gv_region] += result_queue[gv2][:, 0]
                        results_array[:, gv_region + 6] += result_queue[gv2][:, 1]
                        seroprevalence_array[:, gv_region] += result_queue[gv2][:, 2]
                        seroprevalence_array[:, gv_region + 6] += result_queue[gv2][:, 3]
                        integrated_departments.append(gv2)
            workers[gv] = DepartmentRun(params=params.copy(), result_queue=result_queue)
            while threading.active_count() >= max_threads:
                time.sleep(0.00001)
            workers[gv].start()
        current_threads = threading.active_count()
        while len(result_queue) < len(workers):
            while current_threads == threading.active_count() and current_threads > initial_threads and \
                    len(result_queue) < len(workers):
                time.sleep(0.000001)
            if run_type == 'calibration':
                for gv2 in result_queue:
                    if gv2 not in integrated_departments:
                        gv_region = self.regions[gv2] - 1
                        results_array[:, gv_region] += result_queue[gv2][:, 0]
                        results_array[:, gv_region + 6] += result_queue[gv2][:, 1]
                        seroprevalence_array[:, gv_region] += result_queue[gv2][:, 2]
                        seroprevalence_array[:, gv_region + 6] += result_queue[gv2][:, 3]
                        integrated_departments.append(gv2)
            current_threads = threading.active_count()
        if run_type == 'calibration':
            for gv2 in result_queue:
                if gv2 not in integrated_departments:
                    gv_region = self.regions[gv2] - 1
                    results_array[:, gv_region] += result_queue[gv2][:, 0]
                    results_array[:, gv_region + 6] += result_queue[gv2][:, 1]
                    seroprevalence_array[:, gv_region] += result_queue[gv2][:, 2]
                    seroprevalence_array[:, gv_region + 6] += result_queue[gv2][:, 3]
                    integrated_departments.append(gv2)
            for region in range(6):
                results_array[:, region + 12] = seroprevalence_array[:, region] / seroprevalence_array[:, region + 6]
            return results_array
        else:
            pop_pandas = pd.concat(result_queue.values())
            if 'Department' not in exporting_information:
                if 'day' not in exporting_information:
                    exporting_information.append('day')
                pop_pandas = pop_pandas.groupby(exporting_information).sum().reset_index(drop=False)
            pop_pandas['Total_Cost'] = pop_pandas['vac_cost'] + pop_pandas['home_cost'] + pop_pandas['hosp_cost'] + \
                                       pop_pandas['uci_cost']
            pop_pandas['Total_Daly'] = pop_pandas['home_daly'] + pop_pandas['hosp_daly'] + pop_pandas['uci_daly'] + \
                                       pop_pandas['death_daly']
            pop_pandas['Date'] = pop_pandas['day'].apply(lambda x: datetime.datetime(2020, 2, 21) +
                                                                   datetime.timedelta(days=x))
            print('Begin CSV exportation', datetime.datetime.now())
            pop_pandas = pop_pandas.round(decimals=2)
            pop_pandas.to_csv(DIR_OUTPUT + name + '.csv', index=False)
            print('CSV ', DIR_OUTPUT + name + '.csv', 'exported', datetime.datetime.now())
            return pop_pandas


class DepartmentRun(Thread):
    __lock = Lock()

    def __init__(self, params: dict, result_queue: dict):
        Thread.__init__(self)
        self.params = params
        self.result_queue = result_queue

    def run(self):
        # Model Params
        initial_population = self.params['initial_population']
        run_type = self.params['run_type']
        sim_length = self.params['sim_length']
        age_groups, work_groups, health_groups, vaccination_groups = self.params['groups']
        contact_matrix_coefficients = self.params['contact_matrix_coefficients']
        contact_matrix = self.params['contact_matrix']
        max_cm_days = self.params['max_cm_days']
        beta = self.params['beta']
        t_lost_imm = self.params['t_lost_imm']
        arrival_rate = self.params['arrival_rate']
        birth_rate = self.params['birth_rate']
        death_rate = self.params['death_rate']
        morbidity_frac = self.params['morbidity_frac']
        med_degrees = self.params['med_degrees']
        exporting_information = self.params['exporting_information']
        ignore_columns = [item for item in ['Department', 'Health', 'Work', 'Age', 'Vaccine'] if item not in
                          exporting_information]
        gv = self.params['gv']

        # Vaccination run type info
        vaccine_information = self.params.get('vaccine_information', None)
        vaccination_calendar = self.params.get('vaccination_calendar', None)
        vaccination_candidates = ['V0', 'P1', 'S1', 'M1', 'A1'] if run_type == 'vaccination' else None
        vaccine_priority = self.params.get('vaccine_priority', None)
        vaccine_cost = self.params.get('vaccine_cost', None)
        vaccine_start_day, vaccine_end_day = self.params['vaccination_period']

        # All type information
        daly_vector = self.params['daly_vector']
        attention_costs = self.params['attention_costs']

        t_e = self.params['time_params']['t_e']
        t_p = self.params['time_params']['t_p']
        t_sy = self.params['time_params']['t_sy']
        t_a = self.params['time_params']['t_a']
        t_ri = self.params['time_params']['t_ri']
        t_d = self.params['time_params']['t_d']
        t_r = self.params['time_params']['t_r']

        # Prob params
        initial_sus = self.params['prob_params']['initial_sus']
        p_s = self.params['prob_params']['p_s']
        p_c = self.params['prob_params']['p_c']
        p_h = self.params['prob_params']['p_h']
        p_i = self.params['prob_params']['p_i']
        p_c_d = self.params['prob_params']['p_c_d']
        p_h_d = self.params['prob_params']['p_h_d']
        p_i_d = self.params['prob_params']['p_i_d']

        # 0) SU, 1) E, 2) P, 3) SYM, 4) C, 5) HOS, 6) ICU, 7) R_S, 8) A, 9) R, 10) I, 11) Death
        # 12) Cases, 13) Seroprevalence 14) Total_pob 15) New
        i_1_indexes = [2, 3, 4, 7, 10]
        i_2_indexes = [5, 6]
        candidates_indexes = [0, 1, 8, 10]
        alive_compartments = list(range(11))
        percentages = dict()
        if vaccine_priority is not None:
            for phase in vaccine_priority:
                for group in phase:
                    ev, wv, hv, p = group
                    percentages[(ev, wv, hv)] = p
        population = dict()
        covered_percentage = dict()
        for ev in age_groups:
            population_e = dict()
            for wv in work_groups:
                population_w = dict()
                for hv in health_groups:
                    population_h = dict()
                    initial_population_value = initial_population[(ev, wv, hv)]
                    covered_percentage[(ev, wv, hv)] = 1
                    if run_type == 'vaccination':
                        covered_percentage[(ev, wv, hv)] = percentages.get((ev, wv, hv), 0)
                    for vv in vaccination_groups:
                        if vv == 'V0':
                            first_value = initial_population_value * covered_percentage[(ev, wv, hv)]
                        elif vv == 'R':
                            first_value = initial_population_value * (1-covered_percentage[(ev, wv, hv)])
                        else:
                            first_value = 0
                        compartments = list()
                        su = Compartment(name='Susceptible', initial_value=first_value * initial_sus)
                        compartments.append(su)
                        e = Compartment('Exposed')
                        compartments.append(e)
                        p = Compartment('Presymptomatic')
                        compartments.append(p)
                        sy = Compartment('Symptomatic')
                        compartments.append(sy)
                        c = Compartment('Home')
                        compartments.append(c)
                        h = Compartment('Hospitalization')
                        compartments.append(h)
                        i = Compartment('ICU')
                        compartments.append(i)
                        r_s = Compartment('In_recovery')
                        compartments.append(r_s)
                        a = Compartment('Asymptomatic')
                        compartments.append(a)
                        r = Compartment(name='Recovered', initial_value=first_value * (1 - initial_sus))
                        compartments.append(r)
                        imm = Compartment('Immune')
                        compartments.append(imm)
                        d = Compartment('Death')
                        compartments.append(d)
                        cases = Compartment('Cases')
                        compartments.append(cases)
                        seroprevalence = Compartment('Seroprevalence_n')
                        compartments.append(seroprevalence)
                        total_pob = Compartment('total_pob', initial_value=first_value)
                        compartments.append(total_pob)
                        new_home = Compartment('new_home')
                        compartments.append(new_home)
                        new_hosp = Compartment('new_hosp')
                        compartments.append(new_hosp)
                        new_uci = Compartment('new_uci')
                        compartments.append(new_uci)
                        vac_cost = Compartment('vac_cost')
                        compartments.append(vac_cost)
                        home_cost = Compartment('home_cost')
                        compartments.append(home_cost)
                        hosp_cost = Compartment('hosp_cost')
                        compartments.append(hosp_cost)
                        uci_cost = Compartment('uci_cost')
                        compartments.append(uci_cost)
                        home_daly = Compartment('home_daly')
                        compartments.append(home_daly)
                        hosp_daly = Compartment('hosp_daly')
                        compartments.append(hosp_daly)
                        uci_daly = Compartment('uci_daly')
                        compartments.append(uci_daly)
                        death_daly = Compartment('death_daly')
                        compartments.append(death_daly)
                        population_h[vv] = compartments
                    population_w[hv] = population_h
                population_e[wv] = population_w
            population[ev] = population_e
        day = -1
        for t in range(sim_length):
            if day < max_cm_days:
                day += 1
            cur_pob = 0
            for ev in age_groups:
                for wv in work_groups:
                    for hv in health_groups:
                        for vv in vaccination_groups:
                            try:
                                cur_pob += sum(population[ev][wv][hv][vv][state].values[t]
                                               for state in alive_compartments)
                            except Exception as exc:
                                print(ev, wv, hv, vv)
                                print(exc)
                                return
            # Modelo epidemiológico primero
            i_1 = list()
            i_2 = list()
            age_pob = dict()
            for ev in age_groups:
                tot = 0.0
                inf1 = 0.0
                inf2 = 0.0
                for wv in work_groups:
                    for hv in health_groups:
                        for vv in vaccination_groups:
                            tot += sum(population[ev][wv][hv][vv][state].values[t] for state in
                                       alive_compartments)
                            inf1 += sum(population[ev][wv][hv][vv][state].values[t] for state in i_1_indexes)
                            inf2 += sum(population[ev][wv][hv][vv][state].values[t] for state in i_2_indexes)
                            if tot < inf1 + inf2:
                                print(t, gv, ev, wv, hv, vv)
                                print('TOT')
                                for comp in population[ev][wv][hv][vv]:
                                    print(comp.name, comp.values[t])
                                print('i_1')
                                for state in i_1_indexes:
                                    print(population[ev][wv][hv][vv][state].name,
                                          population[ev][wv][hv][vv][state].values[t])
                                print('i_2')
                                for state in i_2_indexes:
                                    print(population[ev][wv][hv][vv][state].name,
                                          population[ev][wv][hv][vv][state].values[t])
                                return None
                cur_pob += tot
                age_pob[ev] = tot
                if tot > 0:
                    i_1.append(inf1 / tot)
                    i_2.append(inf2 / tot)
                    if inf1 + inf2 > tot:
                        print(t, ev, inf1, inf2, inf1 + inf2, tot)
                else:
                    i_1.append(0.0)
                    i_2.append(0.0)
            for ev in age_groups:
                contacts = np.zeros(len(age_groups))
                for cv in contact_matrix:
                    contacts += (1+contact_matrix_coefficients[day][cv]) * np.array(contact_matrix[cv][ev])
                for wv in work_groups:
                    for hv in health_groups:
                        for vv in vaccination_groups:
                            # Bring relevant population
                            su, e, p, sy, c, h, i, r_s, a, r, imm, d, cases, seroprevalence, total_pob, new_home, \
                            new_hosp, new_uci, vac_cost, home_cost, hosp_cost, uci_cost, home_daly, hosp_daly, \
                            uci_daly, death_daly = population[ev][wv][hv][vv]
                            cur_su = su.values.get(t, 0)
                            if cur_su < 0:
                                print('Su', gv, ev, wv, hv, vv, t, cur_su)
                            cur_e = e.values.get(t, 0)
                            if cur_e < 0:
                                print('E', gv, ev, wv, hv, vv, t, cur_e)
                            cur_p = p.values.get(t, 0)
                            if cur_p < 0:
                                print('P', gv, ev, wv, hv, vv, t, cur_p)
                            cur_sy = sy.values.get(t, 0)
                            if cur_sy < 0:
                                print('Sy', gv, ev, wv, hv, vv, t, cur_sy)
                            cur_c = c.values.get(t, 0)
                            if cur_c < 0:
                                print('C', gv, ev, wv, hv, vv, t, cur_c)
                            cur_h = h.values.get(t, 0)
                            if cur_h < 0:
                                print('H', gv, ev, wv, hv, vv, t, cur_h)
                            cur_i = i.values.get(t, 0)
                            if cur_i < 0:
                                print('I', gv, ev, wv, hv, vv, t, cur_i)
                            cur_r_s = r_s.values.get(t, 0)
                            if cur_r_s < 0:
                                print('R_S', gv, ev, wv, hv, vv, t, cur_r_s)
                            cur_a = a.values.get(t, 0)
                            if cur_a < 0:
                                print('A', gv, ev, wv, hv, vv, t, cur_a)
                            cur_r = r.values.get(t, 0)
                            if cur_r < 0:
                                print('R', gv, ev, wv, hv, vv, t, cur_r)
                            cur_imm = imm.values.get(t, 0)
                            if cur_imm < 0:
                                print('Imm', gv, ev, wv, hv, vv, t, cur_imm)
                            cur_d = d.values.get(t, 0)
                            if cur_d < 0:
                                print('D', gv, ev, wv, hv, vv, t, cur_d)
                            cur_cases = cases.values.get(t, 0)
                            if cur_cases < 0:
                                print('Cases', gv, ev, wv, hv, vv, t, cur_cases)
                            cur_seroprevalence = seroprevalence.values.get(t, 0)
                            if cur_seroprevalence < 0:
                                print('Seroprevalence', gv, ev, wv, hv, vv, t, cur_seroprevalence)

                            percent = np.array(i_1) + np.array(i_2) if wv == 'M' else np.array(i_1)
                            percent_change = min(beta * np.dot(percent, contacts), 1.0)
                            contagion_sus = cur_su * percent_change
                            finished_exposed = cur_e / t_e
                            finished_presymptomatic = cur_p / t_p
                            define_treatment = cur_sy / t_sy
                            asymptomatic_recovery = cur_a / t_a
                            home_death_threshold = cur_c / t_d[ev]
                            hosp_death_threshold = cur_h / t_d[ev]
                            icu_death_threshold = cur_i / t_d[ev]
                            new_recovered = cur_r_s/t_r[ev]
                            new_immune = cur_r/t_ri
                            lost_immunity = cur_imm*t_lost_imm if t_lost_imm > 0 else 0
                            arrivals = 0.0
                            if arrival_rate['START_DAY'] <= t <= arrival_rate['END_DAY']:
                                arrivals = arrival_rate['ARRIVAL_RATE']
                                arrivals = arrivals*(total_pob.values.get(t, 0)-d.values.get(t, 0))/cur_pob
                            dsu_dt = {-contagion_sus,
                                      lost_immunity}
                            de_dt = {contagion_sus,
                                       arrivals,
                                       -finished_exposed}
                            dp_dt = {finished_exposed * p_s[(vv, ev)],
                                      -finished_presymptomatic}
                            dsy_dt = {finished_presymptomatic,
                                       -define_treatment}
                            dc_dt = {define_treatment * p_c[(ev, vv, hv)],
                                      -home_death_threshold}
                            dh_dt = {define_treatment * p_h[(ev, vv, hv)],
                                      -hosp_death_threshold}
                            di_dt = {define_treatment * p_i[(ev, vv, hv)],
                                      -icu_death_threshold}
                            dr_s_dt = {home_death_threshold*(1 - p_c_d[(ev, vv, hv)]),
                                        hosp_death_threshold*(1-p_h_d[(ev, vv, hv)]),
                                        icu_death_threshold*(1-p_i_d[(ev, vv, hv)]),
                                        -new_recovered}
                            da_dt = {finished_exposed * (1-p_s[(vv, ev)]),
                                      -asymptomatic_recovery}
                            dr_dt = {new_recovered,
                                      -new_immune}
                            dimm_dt = {asymptomatic_recovery,
                                       new_immune,
                                       -lost_immunity}
                            dd_dt = {home_death_threshold*p_c_d[(ev, vv, hv)],
                                     hosp_death_threshold*p_h_d[(ev, vv, hv)],
                                     icu_death_threshold*p_i_d[(ev, vv, hv)]}
                            if sum(dd_dt) < 0:
                                print('Death decrease', gv, ev, wv, hv, vv, t, 'home', home_death_threshold,
                                      p_c_d[(ev, vv, hv)], 'hosp', hosp_death_threshold, p_h_d[(ev, vv, hv)],
                                      'icu_death_threshold', icu_death_threshold, p_i_d[(ev, vv, hv)])
                            dcases_dt = {finished_exposed * p_s[(vv, ev)]}
                            dseroprevalence_dt = {finished_exposed}

                            su.values[t + 1] = cur_su + float(sum(dsu_dt))
                            e.values[t + 1] = cur_e + float(sum(de_dt))
                            p.values[t + 1] = cur_p + float(sum(dp_dt))
                            sy.values[t + 1] = cur_sy + float(sum(dsy_dt))
                            c.values[t + 1] = cur_c + float(sum(dc_dt))
                            h.values[t + 1] = cur_h + float(sum(dh_dt))
                            i.values[t + 1] = cur_i + float(sum(di_dt))
                            r_s.values[t + 1] = cur_r_s + float(sum(dr_s_dt))
                            a.values[t + 1] = cur_a + float(sum(da_dt))
                            r.values[t + 1] = cur_r + float(sum(dr_dt))
                            imm.values[t + 1] = cur_imm + float(sum(dimm_dt))
                            d.values[t + 1] = cur_d + float(sum(dd_dt))
                            if d.values[t+1] < cur_d:
                                print('Death difference', gv, ev, wv, hv, vv, t, d.values[t + 1], cur_d,
                                      float(sum(dd_dt)))
                            cases.values[t + 1] = cur_cases + float(sum(dcases_dt))
                            seroprevalence.values[t + 1] = cur_seroprevalence + float(sum(dseroprevalence_dt))
                            total_pob.values[t+1] = su.values[t + 1] + e.values[t + 1] + p.values[t + 1] + \
                                                    sy.values[t + 1] + c.values[t + 1] + h.values[t + 1] + \
                                                    i.values[t + 1] + r_s.values[t + 1] + a.values[t + 1] + \
                                                    r.values[t + 1] + imm.values[t + 1] + d.values[t + 1]
                            new_home.values[t + 1] = define_treatment * p_c[(ev, vv, hv)]
                            new_hosp.values[t + 1] = define_treatment * p_h[(ev, vv, hv)]
                            new_uci.values[t + 1] = define_treatment * p_i[(ev, vv, hv)]
                            vac_cost.values[t + 1] = 0
                            if t >=419:
                                home_cost.values[t + 1] = new_home.values[t + 1] * attention_costs[(ev, hv, 'C')]
                                hosp_cost.values[t + 1] = new_hosp.values[t + 1] * attention_costs[(ev, hv, 'H')]
                                uci_cost.values[t + 1] = new_uci.values[t + 1] * attention_costs[(ev, hv, 'I')]
                                home_daly.values[t + 1] = c.values[t + 1] * daly_vector['Home'] / 365 +\
                                                          home_death_threshold*(1 - p_c_d[(ev, vv, hv)]) * \
                                                          daly_vector['Home']*t_r[ev]/365 + \
                                                          r_s.values[t + 1] * daly_vector['InRecovery']/365
                                hosp_daly.values[t + 1] = h.values[t + 1] * daly_vector['Hosp'] / 365 + \
                                                          hosp_death_threshold * (1 - p_h_d[(ev, vv, hv)]) * \
                                                          daly_vector['Hosp'] * t_r[ev] / 365
                                uci_daly.values[t + 1] = i.values[t + 1] * daly_vector['ICU'] / 365 + \
                                                         icu_death_threshold * (1 - p_i_d[(ev, vv, hv)]) * \
                                                          daly_vector['ICU'] * t_r[ev] / 365
                                death_daly.values[t+1] = float(sum(dd_dt)) * daly_vector['Death'][ev]

            # Vaccination dynamics
            if vaccine_start_day <= t <= vaccine_end_day:
                vaccine_capacity = vaccination_calendar[t]
                assignation = calculate_vaccine_assignments(population=population, day=t+1,
                                                                 vaccine_priority=vaccine_priority,
                                                                 vaccine_capacity=vaccine_capacity,
                                                                 candidates_indexes=candidates_indexes)
                for o_vv in vaccination_calendar[t].keys():
                    cur_vaccine_cost = vaccine_cost if o_vv == 'J' else vaccine_cost / 2
                    for ev in age_groups:
                        for wv in work_groups:
                            for hv in health_groups:
                                for i_vv in vaccination_candidates:
                                    # Susceptible:
                                    assigned_s = assignation.get((ev, wv, hv, i_vv, 0, o_vv), 0)
                                    assigned_e = assignation.get((ev, wv, hv, i_vv, 1, o_vv), 0)
                                    assigned_a = assignation.get((ev, wv, hv, i_vv, 8, o_vv), 0)
                                    assigned_im = assignation.get((ev, wv, hv, i_vv, 10, o_vv), 0)
                                    if assigned_s + assigned_e + assigned_a + assigned_im > 0:
                                        d_i_su = {-assigned_s}
                                        d_i_e = {-assigned_e}
                                        d_i_a = {-assigned_a}
                                        d_i_im = {-assigned_im}
                                        d_o_su = {assigned_s*(1-vaccine_information[o_vv[0]]['dose_effectivity'])}
                                        d_o_p = {assigned_e*p_s[(i_vv, ev)]}
                                        d_o_a = {assigned_e*(1-p_s[(i_vv, ev)])}
                                        d_o_im = {assigned_s*vaccine_information[o_vv[0]]['dose_effectivity'],
                                                   assigned_a,
                                                   assigned_im}
                                        population[ev][wv][hv][i_vv][0].values[t + 1] += sum(d_i_su)
                                        population[ev][wv][hv][i_vv][1].values[t + 1] += sum(d_i_e)
                                        population[ev][wv][hv][i_vv][8].values[t + 1] += sum(d_i_a)
                                        population[ev][wv][hv][i_vv][10].values[t + 1] += sum(d_i_im)

                                        population[ev][wv][hv][o_vv][0].values[t + 1] += sum(d_o_su)
                                        population[ev][wv][hv][o_vv][2].values[t + 1] += sum(d_o_p)
                                        population[ev][wv][hv][o_vv][8].values[t + 1] += sum(d_o_a)
                                        population[ev][wv][hv][o_vv][10].values[t + 1] += sum(d_o_im)
                                        population[ev][wv][hv][o_vv][18].values[t + 1] = cur_vaccine_cost * \
                                                                                             (assigned_s +
                                                                                              assigned_e +
                                                                                              assigned_a +
                                                                                              assigned_im)

            # Demographics / Health degrees
            if run_type != 'calibration':
                births = birth_rate*cur_pob
                for state in alive_compartments:
                    previous_growing_o_s = births if state == 0 else 0.0
                    previous_growing_o_h = 0.0
                    previous_growing_m_s = 0.0
                    previous_growing_m_h = 0.0
                    for vv in vaccination_groups:
                        for el in range(len(age_groups)):
                            cur_m_s = population[age_groups[el]]['M']['S'][vv][state].values[t+1]
                            cur_m_h = population[age_groups[el]]['M']['H'][vv][state].values[t+1]
                            cur_o_s = population[age_groups[el]]['O']['S'][vv][state].values[t + 1]
                            cur_o_h = population[age_groups[el]]['O']['H'][vv][state].values[t + 1]
                            growing_m_s = cur_m_s / 1826 if el < len(age_groups)-1 else 0.0
                            growing_m_h = cur_m_h / 1826 if el < len(age_groups)-1 else 0.0
                            growing_o_s = cur_o_s / 1826 if el < len(age_groups) - 1 else 0.0
                            growing_o_h = cur_o_h / 1826 if el < len(age_groups) - 1 else 0.0
                            dying_m_s = cur_m_s*death_rate[age_groups[el]] if state not in \
                                                                                          i_2_indexes else 0.0
                            dying_m_h = cur_m_h * death_rate[age_groups[el]] if state not in \
                                                                                            i_2_indexes else 0.0
                            dying_o_s = cur_o_s * death_rate[age_groups[el]] if state not in \
                                                                                            i_2_indexes else 0.0
                            dying_o_h = cur_o_h * death_rate[age_groups[el]] if state not in \
                                                                                            i_2_indexes else 0.0
                            dm_s_dt = {previous_growing_m_s * (1-morbidity_frac[age_groups[el]]) *
                                       (1-med_degrees[age_groups[el]]['M_TO_O']),
                                       previous_growing_o_s * (1 - morbidity_frac[age_groups[el]]) *
                                       (med_degrees[age_groups[el]]['O_TO_M']),
                                       -growing_m_s,
                                       -dying_m_s}
                            do_s_dt = {previous_growing_o_s * (1 - morbidity_frac[age_groups[el]]) *
                                       (1 - med_degrees[age_groups[el]]['O_TO_M']),
                                       previous_growing_m_s * (1 - morbidity_frac[age_groups[el]]) *
                                       (med_degrees[age_groups[el]]['M_TO_O']),
                                       -growing_o_s,
                                       -dying_o_s}

                            dm_h_dt = {previous_growing_m_s * morbidity_frac[age_groups[el]] *
                                       (1-med_degrees[age_groups[el]]['M_TO_O']),
                                       previous_growing_m_h *
                                       (1 - med_degrees[age_groups[el]]['M_TO_O']),
                                       previous_growing_o_h * (med_degrees[age_groups[el]]['O_TO_M']),
                                       previous_growing_o_s * (1-morbidity_frac[age_groups[el]]) *
                                       (med_degrees[age_groups[el]]['O_TO_M']),
                                        -growing_m_h,
                                        -dying_m_h}

                            do_h_dt = {previous_growing_o_s * morbidity_frac[age_groups[el]] *
                                       (1 - med_degrees[age_groups[el]]['O_TO_M']),
                                       previous_growing_m_h *
                                       (1 - med_degrees[age_groups[el]]['M_TO_O']),
                                       previous_growing_o_h * (med_degrees[age_groups[el]]['O_TO_M']),
                                       previous_growing_m_s * morbidity_frac[age_groups[el]] *
                                       med_degrees[age_groups[el]]['M_TO_O'],
                                       -growing_o_h,
                                       -dying_o_h}

                            population[age_groups[el]]['M']['S'][vv][state].values[t + 1] += sum(dm_s_dt)
                            population[age_groups[el]]['M']['H'][vv][state].values[t + 1] += sum(dm_h_dt)
                            population[age_groups[el]]['O']['S'][vv][state].values[t + 1] += sum(do_s_dt)
                            population[age_groups[el]]['O']['H'][vv][state].values[t + 1] += sum(do_h_dt)

                            previous_growing_o_s = growing_o_s
                            previous_growing_o_h = growing_o_h
                            previous_growing_m_s = growing_m_s
                            previous_growing_m_h = growing_m_h
        if run_type == 'calibration':
            results_array = np.zeros(shape=(sim_length + 1, 4))
            for ev in age_groups:
                for wv in work_groups:
                    for hv in health_groups:
                        for vv in vaccination_groups:
                            results_array[:, 0] += np.array(list(population[ev][wv][hv][vv][12].values.values()),
                                                            dtype=float)  # Cases
                            results_array[:, 1] += np.array(list(population[ev][wv][hv][vv][11].values.values()),
                                                            dtype=float)  # Deaths
                            results_array[:, 2] += np.array(list(population[ev][wv][hv][vv][13].values.values()),
                                                            dtype=float)  # Seroprevalence n
                            results_array[:, 3] += np.array(list(population[ev][wv][hv][vv][14].values.values()),
                                                            dtype=float)  # Total pob
            self.result_queue[gv] = results_array
        else:
            pop_pandas = pd.DataFrame()
            for wv in work_groups:
                pop_pandas_w = pd.DataFrame()
                for hv in health_groups:
                    pop_pandas_h = pd.DataFrame()
                    for vv in vaccination_groups:
                        pop_pandas_v = pd.DataFrame()
                        for ev in age_groups:
                            pop_dict = dict()
                            for comp in population[ev][wv][hv][vv]:
                                pop_dict[comp.name] = comp.values
                            del population[ev][wv][hv][vv]
                            cur_pop_pandas = pd.DataFrame.from_dict(pop_dict).reset_index(drop=False).rename(
                                columns={'index': 'day'})
                            cur_pop_pandas['Age'] = ev
                            if len(pop_pandas_v) > 0:
                                pop_pandas_v = pd.concat([pop_pandas_v, cur_pop_pandas], ignore_index=True)
                            else:
                                pop_pandas_v = cur_pop_pandas.copy()
                            del cur_pop_pandas
                        pop_pandas_v['Vaccine'] = vv
                        if len(pop_pandas_h) > 0:
                            pop_pandas_h = pd.concat([pop_pandas_h, pop_pandas_v], ignore_index=True)
                        else:
                            pop_pandas_h = pop_pandas_v.copy()
                        del pop_pandas_v
                    pop_pandas_h['Health'] = hv
                    if len(pop_pandas_w) > 0:
                        pop_pandas_w = pd.concat([pop_pandas_w, pop_pandas_h], ignore_index=True)
                    else:
                        pop_pandas_w = pop_pandas_h.copy()
                    del pop_pandas_h
                pop_pandas_w['Work'] = wv
                if len(pop_pandas) > 0:
                    pop_pandas = pd.concat([pop_pandas, pop_pandas_w], ignore_index=True)
                else:
                    pop_pandas = pop_pandas_w.copy()
                del pop_pandas_w
            pop_pandas = pop_pandas[pop_pandas['total_pob'] > 0]
            if 'Department' in exporting_information:
                pop_pandas['Department'] = gv
            else:
                ignore_columns.remove('Department')
            if 'day' not in exporting_information:
                exporting_information.append('day')
            if len(ignore_columns) > 0:
                pop_pandas = pop_pandas.drop(columns=ignore_columns)
            pop_pandas = pop_pandas.groupby(exporting_information).sum().reset_index(drop=False)
            self.result_queue[gv] = pop_pandas
