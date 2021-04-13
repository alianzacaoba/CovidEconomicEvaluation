import math
from typing import List, Any
from logic.compartment import Compartment
import pandas as pd
import numpy as np
from tqdm import tqdm
import warnings
import json
from pyexcelerate import Workbook
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


def create_vaccine_assignation(start_date: int, end_date: int, vaccines_information: dict):
    """
    Creates the daily vaccine assignation considering a time period and the required information to determine dosage.
    :param start_date: Starting date of the vaccination program.
    :param end_date: End date of the vaccination program (time to reach the limit).
    :param vaccines_information: Considers the information of each vaccine ordered by name: 1) total_doses considered,
    2) period between first and second dose.
    :return: dictionary with the vaccine time assignation.
    """
    result_information = dict()
    for day in range(start_date, end_date + 1):
        result_information[day] = dict()
    time_period = end_date - start_date + 1
    for v in vaccines_information.keys():
        # v is the name of the vaccine
        vac_info = vaccines_information[v]
        total_number = vac_info['total_doses']
        inter_dose_time = vac_info['inter_dose_time']
        if inter_dose_time == 0:
            daily_number = total_number/time_period
            for day in range(start_date, end_date+1):
                result_information[day][v] = ['V0', daily_number]
        else:
            total_individuals = total_number / 2
            daily_number = total_individuals/(time_period-inter_dose_time)
            for day in range(start_date, end_date + 1 - inter_dose_time):
                result_information[day][v+str(1)] = ['V0', daily_number]
                result_information[day+inter_dose_time][v + str(2)] = [v+str(1), daily_number]
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
            for ev in self.age_groups:
                for wv in self.work_groups:
                    for hv in self.health_groups:
                        self.initial_population[(gv, ev, wv, hv)] = \
                            float(initial_pop[(initial_pop['DEPARTMENT'] == gv) & (initial_pop['AGE_GROUP'] == ev)
                                              & (initial_pop['WORK_GROUP'] == wv) &
                                              (initial_pop['HEALTH_GROUP'] == hv)].POPULATION.sum())
        self.daly_vector = {'Home': {'INF_VALUE': 0.002, 'BASE_VALUE': 0.006, 'MAX_VALUE': 0.012},
                            'Hospital': {'INF_VALUE': 0.032, 'BASE_VALUE': 0.051, 'MAX_VALUE': 0.074},
                            'ICU': {'INF_VALUE': 0.088, 'BASE_VALUE': 0.133, 'MAX_VALUE': 0.190},
                            'Death': {'INF_VALUE': 1, 'BASE_VALUE': 1, 'MAX_VALUE': 1},
                            'Recovered': {'INF_VALUE': 0.0, 'BASE_VALUE': 0.0, 'MAX_VALUE': 0.006}}
        self.vaccine_cost = {'INF_VALUE': 1035.892076, 'BASE_VALUE': 9413.864213, 'MAX_VALUE': 32021.81881}
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
        self.contact_matrix_coefficients = pd.read_csv(DIR_INPUT + 'contact_matrix_coefficients.csv')
        self.max_cm_days = int(self.contact_matrix_coefficients.SIM_DAY.max())
        self.contact_matrix_coefficients.set_index(['SIM_DAY', 'REGIONS'], inplace=True)
        self.contact_matrix_coefficients = self.contact_matrix_coefficients.to_dict(orient='index')

        self.birth_rates = pd.read_csv(DIR_INPUT + 'birth_rate.csv', sep=';', index_col=0).to_dict()['BIRTH_RATE']
        morbidity_frac = pd.read_csv(DIR_INPUT + 'morbidity_fraction.csv', sep=';', index_col=0)
        self.morbidity_frac = morbidity_frac.to_dict()['COMORBIDITY_RISK']
        del morbidity_frac
        self.death_rates = pd.read_csv(DIR_INPUT + 'death_rate.csv', sep=';', index_col=[0, 1]).to_dict()['DEATH_RATE']
        self.med_degrees = pd.read_csv(DIR_INPUT + 'medical_degrees.csv', sep=';',
                                       index_col=[0, 1]).to_dict(orient='index')
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
            vaccine_start_days: dict = None, vaccine_end_days: dict = None, sim_length: int = 365*3,
            export_type: list = None, use_tqdm: bool = False, t_lost_inm: int = 0):

        # run_type:
        #   1) 'calibration': no vaccination status - returns the daily results of measurement (cases, deaths,
        #   seroprevalence)
        #   2) 'vaccination': model with vaccine states - requires the vaccines information characteristics.
        #   3) 'non-vaccination': model without vaccines - [0V] is the only vaccination states
        # vaccine_information debe_tener {'vacuna':{'total_doses', 'inter_dose_time', 'effectivity',
        # 'symptomatic_prob_reduction'}}
        # export_type = {'all', 'json', 'csv', 'xlsx'}
        vaccine_start_day = sim_length +5
        vaccine_end_day = sim_length +6
        if vaccine_start_days is not None and vaccine_end_days is not None:
            vaccine_start_day = vaccine_start_days[type_params['vaccine_day']]
            vaccine_end_day = vaccine_end_days[type_params['vaccine_day']]
        if vaccine_information is not None:
            for vac in vaccine_information:
                vaccine_information[vac]['dose_effectivity'] = 1-math.sqrt(1-vaccine_information[vac]['effectivity'])
        export_type = ['csv'] if export_type is None else export_type
        vaccination_calendar = create_vaccine_assignation(start_date=vaccine_start_day, end_date=vaccine_end_day,
                                                          vaccines_information=vaccine_information) \
            if run_type == 'vaccination' else None
        population = dict()
        departments = self.departments
        age_groups = self.age_groups
        work_groups = self.work_groups
        health_groups = self.health_groups
        vaccination_groups = ['V0', 'P1', 'P2', 'J', 'S1', 'S2', 'M1', 'M2', 'A1', 'A2', 'R'] \
            if run_type == 'vaccination' else ['V0']
        vaccination_candidates = ['V0', 'P1', 'S1', 'M1', 'A1'] if run_type == 'vaccination' else None

        daly_vector = dict()
        for dv in self.daly_vector:
            daly_vector[dv] = self.daly_vector[dv][type_params['daly']]
        t_e = self.time_params[('t_e', 'ALL')][type_params['t_e']]
        t_p = self.time_params[('t_p', 'ALL')][type_params['t_p']]
        t_sy = self.time_params[('t_sy', 'ALL')][type_params['t_sy']]
        t_a = self.time_params[('t_a', 'ALL')][type_params['t_a']]
        t_ri = self.time_params[('t_ri', 'ALL')][type_params['t_ri']]
        initial_sus = self.prob_params[('initial_sus', 'ALL', 'ALL')][type_params['initial_sus']]
        t_d = dict()
        t_r = dict()
        p_s = dict()
        p_c = dict()
        p_h = dict()
        p_i = dict()
        p_c_d = dict()
        p_h_d = dict()
        p_i_d = dict()
        for ev in age_groups:
            t_d[ev] = self.time_params[('t_d', ev)][type_params['t_d']]
            t_r[ev] = max(self.time_params[('t_r', ev)][type_params['t_r']] - t_d[ev], 1)
            p_s[('V0', ev)] = min(self.prob_params[('p_s', ev, 'ALL')][type_params['p_s']]*symptomatic_coefficient, 1.0)
            for hv in health_groups:
                p_c[(ev, 'V0', hv)] = self.prob_params[('p_c', ev, hv)][type_params['p_c']]
                p_h[(ev, 'V0', hv)] = self.prob_params[('p_h', ev, hv)][type_params['p_h']]
                p_i[(ev, 'V0', hv)] = 1 - p_c[(ev, 'V0', hv)] - p_h[(ev, 'V0', hv)]
                for gv in departments:
                    cur_region = self.regions[gv] - 1
                    p_c_d[(gv, ev, 'V0', hv)] = min(death_coefficient[cur_region] *
                                              self.prob_params[('p_c_d', ev, hv)][type_params['p_c_d']], 1.0)
                    p_h_d[(gv, ev, 'V0', hv)] = min(death_coefficient[cur_region] *
                                              self.prob_params[('p_h_d', ev, hv)][type_params['p_h_d']], 1.0)
                    p_i_d[(gv, ev, 'V0', hv)] = min(death_coefficient[cur_region] *
                                              self.prob_params[('p_i_d', ev, hv)][type_params['p_i_d']], 1.0)
        if run_type == 'vaccination':
            for ev in age_groups:
                t_d[ev] = self.time_params[('t_d', ev)][type_params['t_d']]
                t_r[ev] = max(self.time_params[('t_r', ev)][type_params['t_r']] - t_d[ev], 1)
                p_s[('R', ev)] = min(self.prob_params[('p_s', ev, 'ALL')][type_params['p_s']] * symptomatic_coefficient,
                                     1.0)
                for hv in health_groups:
                    p_c[(ev, 'R', hv)] = self.prob_params[('p_c', ev, hv)][type_params['p_c']]
                    p_h[(ev, 'R', hv)] = self.prob_params[('p_h', ev, hv)][type_params['p_h']]
                    p_i[(ev, 'R', hv)] = 1 - p_c[(ev, 'R', hv)] - p_h[(ev, 'R', hv)]
                    for gv in departments:
                        cur_region = self.regions[gv] - 1
                        p_c_d[(gv, ev, 'R', hv)] = min(death_coefficient[cur_region] *
                                                       self.prob_params[('p_c_d', ev, hv)][type_params['p_c_d']], 1.0)
                        p_h_d[(gv, ev, 'R', hv)] = min(death_coefficient[cur_region] *
                                                       self.prob_params[('p_h_d', ev, hv)][type_params['p_h_d']], 1.0)
                        p_i_d[(gv, ev, 'R', hv)] = min(death_coefficient[cur_region] *
                                                       self.prob_params[('p_i_d', ev, hv)][type_params['p_i_d']], 1.0)
            for vv in ['P2', 'J', 'S2', 'M2', 'A2']:  # Half process
                for ev in age_groups:
                    p_s[(vv, ev)] = p_s[('V0', ev)] * (1 - vaccine_information[vv[0]].get('symptomatic_prob_reduction',
                                                                                          0))
            for vv in ['P1', 'S1', 'M1', 'A1']:  # Half process
                for ev in age_groups:
                    p_s[(vv, ev)] = (p_s[('V0', ev)] + p_s[(vv[0] + str(2), ev)]) / 2
            for ev in age_groups:
                for hv in health_groups:
                    for vv in ['P2', 'J', 'S2', 'M2', 'A2']:  # Half process
                        p_i[(ev, vv, hv)] = p_i[(ev, 'V0', hv)] * vaccine_information[vv[0]].get('icu_prob_reduction',
                                                                                                 1)
                        p_h[(ev, vv, hv)] = p_h[(ev, 'V0', hv)] * vaccine_information[vv[0]].get('hosp_prob_reduction',
                                                                                                 1)
                        p_c[(ev, vv, hv)] = 1 - p_i[(ev, vv, hv)] - p_h[(ev, vv, hv)]
                        for gv in departments:
                            p_c_d[(gv, ev, vv, hv)] = p_c_d[(gv, ev, 'V0', hv)] *\
                                                      vaccine_information[vv[0]].get('home_death_reduction', 1)
                            p_h_d[(gv, ev, vv, hv)] = p_h_d[(gv, ev, 'V0', hv)] *\
                                                      vaccine_information[vv[0]].get('hosp_death_reduction', 1)
                            p_i_d[(gv, ev, vv, hv)] = p_i_d[(gv, ev, 'V0', hv)] *\
                                                      vaccine_information[vv[0]].get('icu_death_reduction', 1)
                    for vv in ['P1', 'S1', 'M1', 'A1']:  # Half process
                        p_i[(ev, vv, hv)] = (p_i[(ev, 'V0', hv)]+p_i[(ev, vv[0]+str(2), hv)])/2
                        p_h[(ev, vv, hv)] = (p_h[(ev, 'V0', hv)] + p_h[(ev, vv[0] + str(2), hv)]) / 2
                        p_c[(ev, vv, hv)] = 1 - p_i[(ev, vv, hv)] - p_h[(ev, vv, hv)]
                        for gv in departments:
                            p_c_d[(gv, ev, vv, hv)] = (p_c_d[(gv, ev, 'V0', hv)] + p_c_d[(gv, ev, vv[0]+str(2), hv)])/2
                            p_h_d[(gv, ev, vv, hv)] = (p_h_d[(gv, ev, 'V0', hv)] + p_h_d[(gv, ev, vv[0]+str(2), hv)])/2
                            p_i_d[(gv, ev, vv, hv)] = (p_i_d[(gv, ev, 'V0', hv)] + p_i_d[(gv, ev, vv[0]+str(2), hv)])/2

        # 0) SU, 1) E, 2) P, 3) SYM, 4) C, 5) HOS, 6) ICU, 7) R_S, 8) A, 9) R, 10) I, 11) Death
        # 12) Cases, 13) Seroprevalence 14) Total_pob
        i_1_indexes = [2, 3, 4, 7, 10]
        i_2_indexes = [5, 6]
        candidates_indexes = [0, 1, 8, 10]
        alive_compartments = list(range(12))
        percentages = dict()
        if vaccine_priority is not None:
            for phase in vaccine_priority:
                for group in phase:
                    percentages[group] = phase[group]

        for gv in departments:
            population_g = dict()
            for ev in age_groups:
                population_e = dict()
                for wv in work_groups:
                    population_w = dict()
                    for hv in health_groups:
                        population_h = dict()
                        for vv in vaccination_groups:
                            first_value = self.initial_population[(gv, ev, wv, hv)] if vv in ['V0', 'R'] else 0
                            if run_type == 'vaccination':
                                p = percentages.get((ev, wv, hv), 0)
                                first_value *= p if vv == 'V0' else 1-p
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
                            inm = Compartment('Immune')
                            compartments.append(inm)
                            d = Compartment('Death')
                            compartments.append(d)
                            cases = Compartment('Cases')
                            compartments.append(cases)
                            seroprevalence = Compartment('Seroprevalence_n')
                            compartments.append(seroprevalence)
                            total_pob = Compartment('total_pob', initial_value=first_value)
                            compartments.append(total_pob)
                            population_h[vv] = compartments
                        population_w[hv] = population_h
                    population_e[wv] = population_w
                population_g[ev] = population_e
            population[gv] = population_g

        iterator = tqdm(range(sim_length)) if use_tqdm else range(sim_length)
        for t in iterator:
            dep_pob = dict()
            cur_pob = 0
            for gv in departments:
                for ev in age_groups:
                    for wv in work_groups:
                        for hv in health_groups:
                            for vv in vaccination_groups:
                                try:
                                    cur_pob += sum(population[gv][ev][wv][hv][vv][state].values[t]
                                                   for state in alive_compartments)
                                except Exception as exc:
                                    print(gv, ev, wv, hv, vv)
                                    print(exc)
                                    return

            # Modelo epidemiológico primero
            for gv in departments:
                cur_region = self.regions[gv]-1
                i_1 = list()
                i_2 = list()
                dep_pob[gv] = 0.0
                age_pob = dict()
                for ev in age_groups:
                    tot = 0.0
                    inf1 = 0.0
                    inf2 = 0.0
                    for wv in work_groups:
                        for hv in health_groups:
                            for vv in vaccination_groups:
                                tot += sum(population[gv][ev][wv][hv][vv][state].values[t] for state in
                                           alive_compartments)
                                inf1 += sum(population[gv][ev][wv][hv][vv][state].values[t] for state in i_1_indexes)
                                inf2 += sum(population[gv][ev][wv][hv][vv][state].values[t] for state in i_2_indexes)
                                if tot < inf1 + inf2:
                                    print(t, gv, ev, wv, hv, vv)
                                    print('TOT')
                                    for comp in population[gv][ev][wv][hv][vv]:
                                        print(comp.name, comp.values[t])
                                    print('i_1')
                                    for state in i_1_indexes:
                                        print(population[gv][ev][wv][hv][vv][state].name,
                                              population[gv][ev][wv][hv][vv][state].values[t])
                                    print('i_2')
                                    for state in i_2_indexes:
                                        print(population[gv][ev][wv][hv][vv][state].name,
                                              population[gv][ev][wv][hv][vv][state].values[t])
                                    return None
                    dep_pob[gv] += tot
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
                    for cv in self.contact_matrix:
                        contacts += (1+self.contact_matrix_coefficients[(min(t, self.max_cm_days), cur_region+1)][cv]) \
                                    * np.array(self.contact_matrix[cv][ev])
                    for wv in work_groups:
                        for hv in health_groups:
                            for vv in vaccination_groups:
                                # Bring relevant population
                                su, e, p, sy, c, h, i, r_s, a, r, inm, d, cases, seroprevalence, total_pob \
                                    = population[gv][ev][wv][hv][vv]
                                cur_su = su.values[t]
                                cur_e = e.values[t]
                                cur_p = p.values[t]
                                cur_sy = sy.values[t]
                                cur_c = c.values[t]
                                cur_h = h.values[t]
                                cur_i = i.values[t]
                                cur_r_s = r_s.values[t]
                                cur_a = a.values[t]
                                cur_r = r.values[t]
                                cur_inm = inm.values[t]
                                cur_d = d.values[t]
                                cur_cases = cases.values[t]
                                cur_seroprevalence = seroprevalence.values[t]

                                percent = np.array(i_1) + np.array(i_2) if wv == 'M' else np.array(i_1)
                                percent_change = min(beta[cur_region] * np.dot(percent, contacts), 1.0)
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
                                lost_immunity = cur_inm/t_lost_inm if t_lost_inm > 0 else 0

                                dsu_dt = {-contagion_sus,
                                          lost_immunity}
                                de_dt = {contagion_sus,
                                           ((self.arrival_rate['ARRIVAL_RATE'][gv] * arrival_coefficient[cur_region] *
                                             cur_pob / dep_pob[gv]) if self.arrival_rate['START_DAY'][gv] <= t <=
                                                                       self.arrival_rate['END_DAY'][gv] else 0.0),
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
                                dr_s_dt = {home_death_threshold*(1 - p_c_d[(gv, ev, vv, hv)]),
                                            hosp_death_threshold*(1-p_h_d[(gv, ev, vv, hv)]),
                                            icu_death_threshold*(1-p_i_d[(gv, ev, vv, hv)]),
                                            -new_recovered}
                                da_dt = {finished_exposed * (1-p_s[(vv, ev)]),
                                          -asymptomatic_recovery}
                                dr_dt = {new_recovered,
                                          -new_immune}
                                dinm_dt = {asymptomatic_recovery,
                                           new_immune,
                                           -lost_immunity}
                                dd_dt = {home_death_threshold*p_c_d[(gv, ev, vv, hv)],
                                         hosp_death_threshold*p_h_d[(gv, ev, vv, hv)],
                                         icu_death_threshold*p_i_d[(gv, ev, vv, hv)]}
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
                                inm.values[t + 1] = cur_inm + float(sum(dinm_dt))
                                d.values[t + 1] = cur_d + float(sum(dd_dt))
                                cases.values[t + 1] = cur_cases + float(sum(dcases_dt))
                                seroprevalence.values[t + 1] = cur_seroprevalence + float(sum(dseroprevalence_dt))
                                total_pob.values[t+1] = su.values[t + 1] + e.values[t + 1] + p.values[t + 1] + \
                                                        sy.values[t + 1] + c.values[t + 1] + h.values[t + 1] + \
                                                        i.values[t + 1] + r_s.values[t + 1] + a.values[t + 1] + \
                                                        r.values[t + 1] + inm.values[t + 1] + d.values[t + 1]
            # Vaccination dynamics
            if vaccine_start_day <= t <= vaccine_end_day:
                vaccine_capacity = vaccination_calendar[t]
                assignation = self.calculate_vaccine_assignments(population=population, day=t,
                                                                 vaccine_priority=vaccine_priority,
                                                                 vaccine_capacity=vaccine_capacity,
                                                                 candidates_indexes=candidates_indexes,
                                                                 vaccination_candidates=vaccination_candidates)
                for gv in self.departments:
                    for ev in self.age_groups:
                        for wv in self.work_groups:
                            for hv in self.health_groups:
                                for i_vv in vaccination_candidates:
                                    for o_vv in vaccination_calendar[t].keys():
                                        # Susceptible:
                                        assigned_s = assignation.get([(gv, ev, wv, hv, i_vv, 0, o_vv)], 0)
                                        assigned_e = assignation.get([(gv, ev, wv, hv, i_vv, 1, o_vv)], 0)
                                        assigned_a = assignation.get([(gv, ev, wv, hv, i_vv, 8, o_vv)], 0)
                                        assigned_im = assignation.get([(gv, ev, wv, hv, i_vv, 10, o_vv)], 0)
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
                                            population[gv][ev][wv][hv][i_vv][0].values[t + 1] += sum(d_i_su)
                                            population[gv][ev][wv][hv][i_vv][1].values[t + 1] += sum(d_i_e)
                                            population[gv][ev][wv][hv][i_vv][8].values[t + 1] += sum(d_i_a)
                                            population[gv][ev][wv][hv][i_vv][10].values[t + 1] += sum(d_i_im)

                                            population[gv][ev][wv][hv][o_vv][0].values[t + 1] += sum(d_o_su)
                                            population[gv][ev][wv][hv][o_vv][2].values[t + 1] += sum(d_o_p)
                                            population[gv][ev][wv][hv][o_vv][8].values[t + 1] += sum(d_o_a)
                                            population[gv][ev][wv][hv][o_vv][10].values[t + 1] += sum(d_o_im)

            # Demographics / Health degrees
            if run_type != 'calibration':
                for gv in self.departments:
                    births = self.birth_rates[gv]*dep_pob[gv]
                    for state in alive_compartments:
                        previous_growing_o_s = births if state == 0 else 0.0
                        previous_growing_o_h = 0.0
                        previous_growing_m_s = 0.0
                        previous_growing_m_h = 0.0
                        for vv in vaccination_groups:
                            for el in range(len(age_groups)):
                                cur_m_s = population[gv][age_groups[el]]['M']['S'][vv][state].values[t+1]
                                cur_m_h = population[gv][age_groups[el]]['M']['H'][vv][state].values[t+1]
                                cur_o_s = population[gv][age_groups[el]]['O']['S'][vv][state].values[t + 1]
                                cur_o_h = population[gv][age_groups[el]]['O']['H'][vv][state].values[t + 1]
                                growing_m_s = cur_m_s / 1826 if el < len(age_groups)-1 else 0.0
                                growing_m_h = cur_m_h / 1826 if el < len(age_groups)-1 else 0.0
                                growing_o_s = cur_o_s / 1826 if el < len(age_groups) - 1 else 0.0
                                growing_o_h = cur_o_h / 1826 if el < len(age_groups) - 1 else 0.0
                                dying_m_s = cur_m_s*self.death_rates[(gv, age_groups[el])] if state not in \
                                                                                              i_2_indexes else 0.0
                                dying_m_h = cur_m_h * self.death_rates[(gv, age_groups[el])] if state not in \
                                                                                                i_2_indexes else 0.0
                                dying_o_s = cur_o_s * self.death_rates[(gv, age_groups[el])] if state not in \
                                                                                                i_2_indexes else 0.0
                                dying_o_h = cur_o_h * self.death_rates[(gv, age_groups[el])] if state not in \
                                                                                                i_2_indexes else 0.0
                                dm_s_dt = {previous_growing_m_s * (1-self.morbidity_frac[age_groups[el]]) *
                                           (1-self.med_degrees[(gv, age_groups[el])]['M_TO_O']),
                                           previous_growing_o_s * (1 - self.morbidity_frac[age_groups[el]]) *
                                           (self.med_degrees[(gv, age_groups[el])]['O_TO_M']),
                                           -growing_m_s,
                                           -dying_m_s}
                                do_s_dt = {previous_growing_o_s * (1 - self.morbidity_frac[age_groups[el]]) *
                                           (1 - self.med_degrees[(gv, age_groups[el])]['O_TO_M']),
                                           previous_growing_m_s * (1 - self.morbidity_frac[age_groups[el]]) *
                                           (self.med_degrees[(gv, age_groups[el])]['M_TO_O']),
                                           -growing_o_s,
                                           -dying_o_s}

                                dm_h_dt = {previous_growing_m_s * self.morbidity_frac[age_groups[el]] *
                                           (1-self.med_degrees[(gv, age_groups[el])]['M_TO_O']),
                                           previous_growing_m_h *
                                           (1 - self.med_degrees[(gv, age_groups[el])]['M_TO_O']),
                                           previous_growing_o_h * (self.med_degrees[(gv, age_groups[el])]['O_TO_M']),
                                           previous_growing_o_s * (1-self.morbidity_frac[age_groups[el]]) *
                                           (self.med_degrees[(gv, age_groups[el])]['O_TO_M']),
                                            -growing_m_h,
                                            -dying_m_h}

                                do_h_dt = {previous_growing_o_s * self.morbidity_frac[age_groups[el]] *
                                           (1 - self.med_degrees[(gv, age_groups[el])]['O_TO_M']),
                                           previous_growing_m_h *
                                           (1 - self.med_degrees[(gv, age_groups[el])]['M_TO_O']),
                                           previous_growing_o_h * (self.med_degrees[(gv, age_groups[el])]['O_TO_M']),
                                           previous_growing_m_s * self.morbidity_frac[age_groups[el]] *
                                           self.med_degrees[(gv, age_groups[el])]['M_TO_O'],
                                           -growing_o_h,
                                           -dying_o_h}

                                population[gv][age_groups[el]]['M']['S'][vv][state].values[t + 1] += sum(dm_s_dt)
                                population[gv][age_groups[el]]['M']['H'][vv][state].values[t + 1] += sum(dm_h_dt)
                                population[gv][age_groups[el]]['O']['S'][vv][state].values[t + 1] += sum(do_s_dt)
                                population[gv][age_groups[el]]['O']['H'][vv][state].values[t + 1] += sum(do_h_dt)

                                previous_growing_o_s = growing_o_s
                                previous_growing_o_h = growing_o_h
                                previous_growing_m_s = growing_m_s
                                previous_growing_m_h = growing_m_h
        if run_type == 'calibration':
            results_array = np.zeros(shape=(sim_length + 1, 18))
            seroprevalence = dict()
            total_pob = dict()
            for gv in departments:
                cur_region = self.regions[gv] - 1

                for ev in age_groups:
                    for wv in work_groups:
                        for hv in health_groups:
                            for vv in vaccination_groups:
                                results_array[:, cur_region] += np.array(list(population[gv][ev][wv][hv][vv][12]
                                                                              .values.values()),
                                                                         dtype=float)  # Cases
                                results_array[:, cur_region+6] += np.array(list(population[gv][ev][wv][hv][vv][11]
                                                                                .values.values()),
                                                                           dtype=float)  # Deaths
                                seroprevalence[cur_region] = seroprevalence.get(cur_region, 0) + \
                                                             np.array(list(population[gv][ev][wv][hv][vv][13]
                                                                           .values.values()), dtype=float)
                                # Seroprevalence n
                                total_pob[cur_region] = total_pob.get(cur_region, 0) + \
                                                        np.array(list(population[gv][ev][wv][hv][vv][14]
                                                                      .values.values()), dtype=float)
            for cur_region in seroprevalence.keys():

                try:
                    results_array[:, cur_region+12] = seroprevalence[cur_region]/total_pob[cur_region]
                except Exception as exc:
                    print(exc)
                    print(cur_region)
                    print(seroprevalence[cur_region])
                    print(total_pob[cur_region])
            return results_array
        else:
            pop_pandas = pd.DataFrame()
            pop_dict = dict()
            print('Consolidating results...')
            for gv in departments:
                pop_dict_g = dict()
                pop_pandas_g = pd.DataFrame()
                for ev in age_groups:
                    pop_dict_e = dict()
                    pop_pandas_e = pd.DataFrame()
                    for wv in work_groups:
                        pop_dict_w = dict()
                        pop_pandas_w = pd.DataFrame()
                        for hv in health_groups:
                            pop_dict_h = dict()
                            pop_pandas_h = dict()
                            for vv in vaccination_groups:
                                pop_dict_v = dict()
                                for comp in population[gv][ev][wv][hv][vv]:
                                    pop_dict_v[comp.name] = comp.values
                                cur_pop_pandas = pd.DataFrame.from_dict(pop_dict_v).reset_index(drop=False).rename(
                                    columns={'index': 'day'})
                                if len(pop_pandas_h) > 0:
                                    pop_pandas_h = pd.concat([pop_pandas_h, cur_pop_pandas], ignore_index=True)
                                else:
                                    pop_pandas_h = cur_pop_pandas.copy()
                                del cur_pop_pandas
                            pop_pandas_h['Health'] = hv
                            if len(pop_pandas_w) > 0:
                                pop_pandas_w = pd.concat([pop_pandas_w, pop_pandas_h], ignore_index=True)
                            else:
                                pop_pandas_w = pop_pandas_h.copy()
                            del pop_pandas_h
                            pop_dict_w[hv] = pop_dict_h
                        pop_pandas_w['Work'] = wv
                        if len(pop_pandas_e) > 0:
                            pop_pandas_e = pd.concat([pop_pandas_e, pop_pandas_w], ignore_index=True)
                        else:
                            pop_pandas_e = pop_pandas_w.copy()
                        del pop_pandas_w
                        pop_dict_e[wv] = pop_dict_w
                    pop_pandas_e['Age'] = ev
                    if len(pop_pandas_g) > 0:
                        pop_pandas_g = pd.concat([pop_pandas_g, pop_pandas_e], ignore_index=True)
                    else:
                        pop_pandas_g = pop_pandas_e.copy()
                    del pop_pandas_e
                    pop_dict_g[ev] = pop_dict_e
                pop_pandas_g['Department'] = gv
                if len(pop_pandas) > 0:
                    pop_pandas = pd.concat([pop_pandas, pop_pandas_g], ignore_index=True)
                else:
                    pop_pandas = pop_pandas_g.copy()
                del pop_pandas_g
                pop_dict[gv] = pop_dict_g
            print('Begin exportation')
            if 'all' in export_type or 'json' in export_type:
                print('Begin JSon exportation')
                with open(DIR_OUTPUT + 'result_' + name + '.json', 'w') as fp:
                    json.dump(pop_dict, fp)
                print('JSon ', DIR_OUTPUT + 'result_' + name + '.json', 'exported')
            if 'all' in export_type or 'csv' in export_type or 'xlsx' in export_type:
                pop_pandas = pop_pandas.set_index(['day', 'Department', 'Health', 'Work', 'Age']).reset_index(
                    drop=False)
                if 'all' in export_type or 'csv' in export_type:
                    print('Begin CSV exportation')
                    pop_pandas.to_csv(DIR_OUTPUT + 'result_' + name + '.csv', index=False)
                    print('CSV ', DIR_OUTPUT + 'result_' + name + '.csv', 'exported')
                if 'all' in export_type or 'xlsx' in export_type:
                    print('Begin excel exportation')
                    wb = Workbook()
                    for gv in tqdm(departments):
                        pop_pandas_current = pop_pandas[pop_pandas['Department'] == gv].drop(columns='Department')
                        pop_pandas_current = pop_pandas_current.set_index(['day', 'Health', 'Work', 'Age']).reset_index(
                            drop=False)
                        values = [pop_pandas_current.columns] + list(pop_pandas_current.values)
                        name_gv = gv if len(gv) < 31 else gv[:15]
                        wb.new_sheet(name_gv, data=values)
                    pop_pandas.drop(columns='Department', inplace=True)
                    pop_pandas_current = pop_pandas.groupby(['day', 'Health', 'Work', 'Age']).sum().reset_index(
                        drop=False)
                    values = [pop_pandas_current.columns] + list(pop_pandas_current.values)
                    print('Excel exportation country results')
                    wb.new_sheet('Country_results', data=values)
                    print('Saving excel results', DIR_OUTPUT + 'result_' + name + '.xlsx')
                    wb.save(DIR_OUTPUT + 'result_' + name + '.xlsx')
                    print('Excel ', DIR_OUTPUT + 'result_' + name + '.xlsx', 'exported')
            return pop_pandas

    def calculate_vaccine_assignments(self, population: dict, day: int, vaccine_priority: list, vaccine_capacity: dict,
                                      candidates_indexes: list, vaccination_candidates: list):
        """
        Calculates the daily vaccination assignments for every group.
        :param population: Resulting population in the simulation.
        :param day: Day to obtain information
        :param vaccine_priority: list of priorities. Each group contains a list of tuples indicating the groups that
        comprehend the priorities.
        :param vaccine_capacity: dictionary with amount for each type of vaccine and target group.
        :param candidates_indexes: indexes for states (infection related) that are candidates to the vaccine.
        :param vaccination_candidates: States that are candidates to be vaccinated
        :return:
        """
        day_population = dict()
        total_candidates = dict()
        assignation = dict()
        for gv in self.departments:
            for vv in vaccination_candidates:
                total_candidates[(gv, vv)] = 0
                for ev in self.age_groups:
                    for wv in self.work_groups:
                        for hv in self.health_groups:
                            for cv in candidates_indexes:
                                day_population[(gv, ev, wv, hv, vv, cv)] = population[gv][ev][wv][hv][vv][cv][day]
                                total_candidates[(gv, vv)] += day_population[(gv, ev, wv, hv, vv, cv)]

        for vaccine in vaccine_capacity:
            # Diccionario con el siguiente orden: {'cual':[a_quien, cantidad]}
            current_vac = vaccine_capacity[vaccine]
            vv = current_vac[0]
            vaccine_candidates = sum(total_candidates[(gv, vv)] for gv in self.departments)
            vaccines_to_use = min(vaccine_candidates, current_vac[1])
            for gv in self.departments:
                vaccines_for_department = vaccines_to_use*total_candidates[(gv, vv)]/vaccine_candidates
                total_candidates[(gv, vv)] -= vaccines_for_department
                # Van en el orden de las listas
                for groups in vaccine_priority:
                    # group = [[ev, wv, hv]]
                    group_candidates = 0
                    for group in groups:
                        ev, wv, hv = group
                        group_candidates += sum(day_population[(gv, ev, wv, hv, vv, cv)] for cv in candidates_indexes)
                    vaccines_for_group = min(vaccines_for_department, group_candidates)
                    vaccines_for_department -= vaccines_for_group
                    for group in groups:
                        ev, wv, hv = group
                        for cv in candidates_indexes:
                            assignation[(gv, ev, wv, hv, vv, cv, vaccine)] = day_population[(gv, ev, wv, hv, vv, cv)] /\
                                                                             vaccines_for_group
                            day_population[(gv, ev, wv, hv, vv, cv)] -= assignation[(gv, ev, wv, hv, vv, cv)]
        return assignation
