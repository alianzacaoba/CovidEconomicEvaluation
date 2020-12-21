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

def calculate_vaccine_assignments(department_population: dict, day: int, vaccine_priority: list,
                                  vaccine_capacity: float, candidates_indexes: list, non_candidates_indexes: list):
    remaining_vaccines = vaccine_capacity
    assignation = dict()
    for groups in vaccine_priority:
        if remaining_vaccines > 0.0:
            candidates = list()
            for group in groups:
                ev, wv, hv, percent = group
                if percent == 1:
                    candidates[(ev,wv,hv)] = sum(department_population[ev][wv][hv][cv].values[day]
                                                 for cv in candidates_indexes)
                else:
                    candidates[(ev,wv,hv)] = max(percent*sum(department_population[ev][wv][hv][cv].values[day]
                                                             for cv in candidates_indexes)
                                                 -(percent-1) * sum(department_population[ev][wv][hv][cv].values[day]
                                                     for cv in non_candidates_indexes),0.0)
            given_vaccines = min(remaining_vaccines, sum(candidates.values()))
            remaining_vaccines -= given_vaccines
            for group in groups:
                ev, wv, hv, percent = group
                assignation[(ev,wv,hv)] = given_vaccines*candidates[(ev,wv,hv)]/\
                                          (sum(candidates.values())*
                                           sum(department_population[ev][wv][hv][cv].values[day]
                                               for cv in candidates_indexes))
        else:
            return assignation
    return assignation


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
            symptomatic_coefficient: float = spc_f, vaccine_priority: list = None, vaccine_capacities: dict = None,
            vaccine_effectiveness: dict = None, vaccine_start_day: dict = None, vaccine_end_day: dict = None,
            sim_length: int = 365*3, export_type: list = None, use_tqdm: bool = False):

        # run_type:
        #   1) 'calibration': for calibration purposes, states f1,f2,v1,v2,e_f,a_f do not exist
        #   2) 'vaccination': model with vaccine states
        #   3) 'non-vaccination': model
        # SU, E, A, R_A, P, Sy, C, H, I, R, D, Cases
        # export_type = {'all', 'json', 'csv', 'xlsx'}
        export_type = ['csv'] if export_type is None else export_type

        population = dict()
        departments = self.departments
        age_groups = self.age_groups
        work_groups = self.work_groups
        health_groups = self.health_groups
        daly_vector = dict()
        for dv in self.daly_vector:
            daly_vector[dv] = self.daly_vector[dv][type_params['daly']]
        t_e = self.time_params[('t_e', 'ALL')][type_params['t_e']]
        t_p = self.time_params[('t_p', 'ALL')][type_params['t_p']]
        t_sy = self.time_params[('t_sy', 'ALL')][type_params['t_sy']]
        t_a = self.time_params[('t_a', 'ALL')][type_params['t_a']]
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
            t_r[ev] = self.time_params[('t_r', ev)][type_params['t_r']]
            p_s[ev] = min(self.prob_params[('p_s', ev, 'ALL')][type_params['p_s']]*symptomatic_coefficient, 1.0)
            for hv in health_groups:
                p_c[(ev, hv)] = self.prob_params[('p_c', ev, hv)][type_params['p_c']]
                p_h[(ev, hv)] = self.prob_params[('p_h', ev, hv)][type_params['p_h']]
                p_i[(ev, hv)] = 1 - p_c[(ev, hv)] - p_h[(ev, hv)]
                for gv in departments:
                    cur_region = self.regions[gv] - 1
                    p_c_d[(gv, ev, hv)] = min(death_coefficient[cur_region] *
                                              self.prob_params[('p_c_d', ev, hv)][type_params['p_c_d']], 1.0)
                    p_h_d[(gv, ev, hv)] = min(death_coefficient[cur_region] *
                                              self.prob_params[('p_h_d', ev, hv)][type_params['p_h_d']], 1.0)
                    p_i_d[(gv, ev, hv)] = min(death_coefficient[cur_region] *
                                              self.prob_params[('p_i_d', ev, hv)][type_params['p_i_d']], 1.0)

        # 0) SU, 1) E, 2) A, 3) R_A, 4) P, 5) Sy, 6) C, 7) R_C, 8) H, 9) R_H, 10)I, 11) R_I 12) R, 13) D, 14)Cases,
        # 15) F1, 16) F2, 17) V1, 18) V2, 19) EF, 20) AF  15(NV)-21(V)) Home cases
        i_1_indexes = [2, 4, 5, 20] if run_type == 'vaccination' else [2, 4, 5]
        i_2_indexes = [6, 7, 10]
        candidates_indexes = [0, 1, 2, 3, 15, 17] if run_type == 'vaccination' else [0, 1, 2, 3]
        alive_compartments = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 15, 16, 17, 18, 19, 20] \
            if run_type == 'vaccination' else [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
        non_candidate_indexes = list(set(alive_compartments)-set(candidates_indexes))
        for gv in departments:
            population_g = dict()
            for ev in age_groups:
                population_e = dict()
                for wv in work_groups:
                    population_w = dict()
                    for hv in health_groups:
                        first_value = self.initial_population[(gv, ev, wv, hv)]
                        compartments = list()
                        su = Compartment(name='Susceptible', initial_value=first_value * initial_sus)
                        compartments.append(su)
                        e = Compartment('Exposed')
                        compartments.append(e)
                        a = Compartment('Asymptomatic')
                        compartments.append(a)
                        r_a = Compartment(name='Recovered_Asymptomatic', initial_value=first_value * (1 - initial_sus))
                        compartments.append(r_a)
                        p = Compartment('Presymptomatic')
                        compartments.append(p)
                        sy = Compartment('Symptomatic')
                        compartments.append(sy)
                        c = Compartment('Home')
                        compartments.append(c)
                        r_c = Compartment('Future_Recovered_Home')
                        compartments.append(r_c)
                        h = Compartment('Hospitalization')
                        compartments.append(h)
                        r_h = Compartment('Future_Recovered_Hospitalization')
                        compartments.append(r_h)
                        i = Compartment('ICU')
                        compartments.append(i)
                        r_i = Compartment('Future_Recovered_ICU')
                        compartments.append(r_i)
                        r = Compartment('Recovered')
                        compartments.append(r)
                        d = Compartment('Death')
                        compartments.append(d)
                        cases = Compartment('Cases')
                        compartments.append(cases)
                        if run_type == 'vaccination':
                            # SU, E, A, R_A, P, Sy, C, H, I, R, D, Cases, F1, F2, V1, V2, EF, AF
                            f_1 = Compartment('Failure_1')
                            compartments.append(f_1)
                            f_2 = Compartment('Failure_2')
                            compartments.append(f_2)
                            v_1 = Compartment('Vaccinated_1')
                            compartments.append(v_1)
                            v_2 = Compartment('Vaccinated_2')
                            compartments.append(v_2)
                            e_f = Compartment('Exposed_Failure')
                            compartments.append(e_f)
                            a_f = Compartment('Asymptomatic_Failure')
                            compartments.append(a_f)
                            day_vaccines = Compartment('Day_Vaccines')
                            compartments.append(day_vaccines)
                        home_cases = Compartment('Home_Cases')
                        compartments.append(home_cases)
                        home_treatment_costs = Compartment('Home_Treatment_Costs')
                        compartments.append(home_treatment_costs)
                        hospital_treatment_costs = Compartment('Hospital_Treatment_Costs')
                        compartments.append(hospital_treatment_costs)
                        icu_treatment_costs = Compartment('ICU_Treatment_Costs')
                        compartments.append(icu_treatment_costs)
                        vaccine_costs = Compartment('Vaccine_Costs')
                        compartments.append(vaccine_costs)
                        total_costs = Compartment('Total_Costs')
                        compartments.append(total_costs)
                        total_daly = Compartment('Total_Daly')
                        compartments.append(total_daly)
                        population_w[hv] = compartments
                    population_e[wv] = population_w
                population_g[ev] = population_e
            population[gv] = population_g
        iterator = tqdm(range(sim_length)) if use_tqdm else range(sim_length)
        for t in iterator:
            dep_pob = dict()
            for gv in departments:
                cur_region = self.regions[gv]-1
                if run_type == 'vaccination' and vaccine_start_day[type_params['vaccine_day']] <= t <= \
                        vaccine_end_day[type_params['vaccine_day']]:
                    vaccine_assignments = calculate_vaccine_assignments(department_population=population[gv],
                                                                        day=t, vaccine_priority=vaccine_priority,
                                                                        vaccine_capacity=float(vaccine_capacities[gv]),
                                                                        candidates_indexes=candidates_indexes,
                                                                        non_candidates_indexes=non_candidate_indexes)
                else:
                    vaccine_assignments = None
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
                            tot += sum(population[gv][ev][wv][hv][state].values[t] for state in alive_compartments)
                            inf1 += sum(population[gv][ev][wv][hv][state].values[t] for state in i_1_indexes)
                            inf2 += sum(population[gv][ev][wv][hv][state].values[t] for state in i_2_indexes)
                            if tot < inf1 + inf2:
                                print(t, gv, ev, wv, hv)
                                print('TOT')
                                for comp in population[gv][ev][wv][hv]:
                                    print(comp.name, comp.values[t])
                                print('i_1')
                                for state in i_1_indexes:
                                    print(population[gv][ev][wv][hv][state].name,
                                          population[gv][ev][wv][hv][state].values[t])
                                print('i_2')
                                for state in i_2_indexes:
                                    print(population[gv][ev][wv][hv][state].name,
                                          population[gv][ev][wv][hv][state].values[t])
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
                            if run_type == 'vaccination':
                                # Bring relevant population
                                su, e, a, r_a, p, sy, c, r_c, h, r_h, i, r_i, r, d, cases, f_1, f_2, v_1, v_2, e_f, \
                                    a_f, day_v, h_c, c_t_c, h_t_c, i_t_c, v_c, t_c, daly = population[gv][ev][wv][hv]

                                cur_su = su.values[t]
                                cur_e = e.values[t]
                                cur_a = a.values[t]
                                cur_r_a = r_a.values[t]
                                cur_p = p.values[t]
                                cur_sy = sy.values[t]
                                cur_c = c.values[t]
                                cur_r_c = r_c.values[t]
                                cur_h = h.values[t]
                                cur_r_h = r_h.values[t]
                                cur_i = i.values[t]
                                cur_r_i = r_i.values[t]
                                cur_r = r.values[t]
                                cur_d = d.values[t]
                                cur_cases = cases.values[t]
                                cur_f_1 = f_1.values[t]
                                cur_f_2 = f_2.values[t]
                                cur_v_1 = v_1.values[t]
                                cur_v_2 = v_2.values[t]
                                cur_e_f = e_f.values[t]
                                cur_a_f = a_f.values[t]
                                cur_pob = cur_su + cur_e + cur_a + cur_r_a + cur_p + cur_sy + cur_c + cur_r_c + cur_h +\
                                          cur_r_h + cur_i + cur_r_i + cur_r + cur_f_1 + cur_f_2 + cur_v_1 + cur_v_2 + \
                                          cur_e_f + cur_a_f
                                # Run vaccination
                                day_v.values[t + 1] = 0.0
                                v_c.values[t + 1] = 0.0

                                if vaccine_assignments is not None:
                                    dsu_dt = {-cur_su * vaccine_assignments.get((ev, wv, hv), 0)}
                                    df_1_dt = {-cur_f_1 * vaccine_assignments.get((ev, wv, hv), 0),
                                               cur_su * (1 - vaccine_effectiveness[(ev, hv)]['VACCINE_EFFECTIVENESS_1'])
                                               * vaccine_assignments.get((ev, wv, hv), 0)
                                               }
                                    df_2_dt = {cur_f_1 * (1 - vaccine_effectiveness[(ev, hv)]['VACCINE_EFFECTIVENESS_2']
                                                          ) * vaccine_assignments.get((ev, wv, hv), 0)
                                               }
                                    de_dt = {-cur_e * vaccine_assignments.get((ev, wv, hv), 0)}
                                    da_dt = {-cur_a * vaccine_assignments.get((ev, wv, hv), 0)}
                                    da_f_dt = {cur_e*(1-p_s[ev])*vaccine_assignments.get((ev, wv, hv), 0)}
                                    dr_a_dt = {-cur_r_a*vaccine_assignments.get((ev, wv, hv), 0)}
                                    dv_1_dt = {-cur_v_1*vaccine_assignments.get((ev, wv, hv), 0),
                                               vaccine_effectiveness[(ev, hv)]['VACCINE_EFFECTIVENESS_1']
                                               * vaccine_assignments.get((ev, wv, hv), 0),
                                               cur_a*vaccine_assignments.get((ev, wv, hv), 0),
                                               cur_r_a*vaccine_assignments.get((ev, wv, hv), 0)
                                               }
                                    dv_2_dt = {cur_v_1*vaccine_assignments.get((ev, wv, hv), 0),
                                               cur_f_1*vaccine_effectiveness[(ev, hv)]['VACCINE_EFFECTIVENESS_2']
                                               * vaccine_assignments.get((ev, wv, hv), 0)
                                               }
                                    dd_v_dt = {cur_su * vaccine_assignments.get((ev, wv, hv), 0),
                                               cur_f_1 * vaccine_assignments.get((ev, wv, hv), 0),
                                               cur_e * vaccine_assignments.get((ev, wv, hv), 0),
                                               cur_a * vaccine_assignments.get((ev, wv, hv), 0),
                                               cur_r_a * vaccine_assignments.get((ev, wv, hv), 0),
                                               cur_v_1 * vaccine_assignments.get((ev, wv, hv), 0)
                                               }

                                    day_v.values[t + 1] = sum(dd_v_dt)
                                    v_c.values[t + 1] = day_v.values[t + 1]*self.vaccine_cost[type_params['cost']]
                                    cur_su += sum(dsu_dt)
                                    cur_f_1 += sum(df_1_dt)
                                    cur_f_2 += sum(df_2_dt)
                                    cur_e += sum(de_dt)
                                    cur_a += sum(da_dt)
                                    cur_a_f += sum(da_f_dt)
                                    cur_r_a += sum(dr_a_dt)
                                    cur_v_1 += sum(dv_1_dt)
                                    cur_v_2 += sum(dv_2_dt)

                                percent = np.array(i_1) + np.array(i_2) if wv == 'M' else np.array(i_1)
                                percent_change = min(beta[cur_region] * np.dot(percent, contacts), 1.0)
                                contagion_sus = cur_su * percent_change
                                contagion_f_1 = cur_f_1 * percent_change
                                contagion_f_2 = cur_f_2 * percent_change
                                dsu_dt = {-contagion_sus}
                                df_1_dt = {-contagion_f_1}
                                df_2_dt = {-contagion_f_2}
                                de_dt = {contagion_sus,
                                         ((self.arrival_rate['ARRIVAL_RATE'][gv] * arrival_coefficient[cur_region] *
                                           cur_pob / dep_pob[gv]) if self.arrival_rate['START_DAY'][gv] <= t <=
                                                                     self.arrival_rate['END_DAY'][gv] else 0.0),
                                         -cur_e / t_e
                                         }
                                de_f_dt = {contagion_f_1,
                                           contagion_f_2,
                                           -cur_e_f/t_e}
                                da_dt = {cur_e * (1 - p_s[ev]) / t_e,
                                         -cur_a / t_a
                                         }
                                da_f_dt = {cur_e_f * (1 - p_s[ev]) / t_e,
                                           -cur_a_f / t_a
                                           }
                                dr_a_dt = {cur_a / t_a
                                           }
                                dp_dt = {cur_e * p_s[ev] / t_e,
                                         cur_e_f * p_s[ev] / t_e,
                                         -cur_p / t_p
                                         }
                                dsy_dt = {cur_p / t_p,
                                          -cur_sy / t_sy
                                          }
                                dc_dt = {cur_sy * p_c[(ev, hv)] / t_sy,
                                         -cur_c / t_d[ev]
                                         }
                                dr_c_dt = {cur_c * (1 - p_c_d[(gv, ev, hv)]) / t_d[ev],
                                           -cur_r_c / (t_r[ev] - t_d[ev])
                                           }
                                dh_dt = {cur_sy * p_h[(ev, hv)] / t_sy,
                                         -cur_h / t_d[ev]
                                         }
                                dr_h_dt = {cur_h * (1 - p_h_d[(gv, ev, hv)]) / t_d[ev],
                                           -cur_r_h / (t_r[ev] - t_d[ev])
                                           }
                                di_dt = {cur_sy * p_i[(ev, hv)] / t_sy,
                                         -cur_i / t_d[ev]
                                         }
                                dr_i_dt = {cur_i * (1 - p_i_d[(gv, ev, hv)]) / t_d[ev],
                                           -cur_r_i / (t_r[ev] - t_d[ev])
                                           }
                                dr_dt = {cur_r_c / (t_r[ev] - t_d[ev]),
                                         cur_r_h / (t_r[ev] - t_d[ev]),
                                         cur_r_i / (t_r[ev] - t_d[ev])
                                         }
                                dd_dt = {cur_c * p_c_d[(gv, ev, hv)] / t_d[ev],
                                         cur_h * p_h_d[(gv, ev, hv)] / t_d[ev],
                                         cur_i * p_i_d[(gv, ev, hv)] / t_d[ev]
                                         }
                                dcases_dt = {cur_e * p_s[ev] / t_e}
                                dv_2_dt = {cur_a_f/t_a}

                                h_c.values[t + 1] = cur_sy * p_h[(ev, hv)] / t_sy
                                su.values[t+1] = cur_su + float(sum(dsu_dt))
                                e.values[t+1] = cur_e + float(sum(de_dt))
                                a.values[t+1] = cur_a + float(sum(da_dt))
                                r_a.values[t+1] = cur_r_a + float(sum(dr_a_dt))
                                p.values[t+1] = cur_p + float(sum(dp_dt))
                                sy.values[t+1] = cur_sy + float(sum(dsy_dt))
                                c.values[t+1] = cur_c + float(sum(dc_dt))
                                r_c.values[t+1] = cur_r_c + float(sum(dr_c_dt))
                                h.values[t+1] = cur_h + float(sum(dh_dt))
                                r_h.values[t+1] = cur_r_h + float(sum(dr_h_dt))
                                i.values[t+1] = cur_i + float(sum(di_dt))
                                r_i.values[t+1] = cur_r_i + float(sum(dr_i_dt))
                                r.values[t+1] = cur_r + float(sum(dr_dt))
                                d.values[t+1] = cur_d + float(sum(dd_dt))
                                cases.values[t+1] = cur_cases + float(sum(dcases_dt))
                                f_1.values[t+1] = cur_f_1 + sum(df_1_dt)
                                f_2.values[t+1] = cur_f_2 + sum(df_2_dt)
                                e_f.values[t+1] = cur_e_f + sum(de_f_dt)
                                a_f.values[t+1] = cur_a_f + sum(da_f_dt)
                                v_2.values[t+1] = cur_v_2 + sum(dv_2_dt)
                                v_1.values[t + 1] = cur_v_1

                                c_t_c.values[t + 1] = h_c.values[t + 1] * \
                                                      self.attention_costs[(ev, hv, 'C')][type_params['cost']]
                                h_t_c.values[t + 1] = (h.values[t + 1]+r_h.values[t + 1]) * \
                                                      self.attention_costs[(ev, hv, 'H')][type_params['cost']]
                                i_t_c.values[t + 1] = (i.values[t + 1]+r_i.values[t + 1]) * \
                                                      self.attention_costs[(ev, hv, 'I')][type_params['cost']]
                                t_c.values[t + 1] = v_c.values[t + 1] + c_t_c.values[t + 1] + h_t_c.values[t + 1] + \
                                                    i_t_c.values[t + 1]
                                daly.values[t+1] = sum([(c.values[t+1] + r_c.values[t+1])*daly_vector['Home'],
                                                       (h.values[t + 1]+r_h.values[t + 1])*daly_vector['Hospital'],
                                                       (i.values[t + 1]+r_i.values[t + 1])*daly_vector['ICU'],
                                                       (r.values[t + 1]) * daly_vector['Recovered'],
                                                       (d.values[t + 1]) * daly_vector['Death']])

                            else:
                                su, e, a, r_a, p, sy, c, r_c, h, r_h, i, r_i, r, d, cases, h_c, c_t_c, h_t_c, i_t_c, \
                                    v_c, t_c, daly = population[gv][ev][wv][hv]
                                cur_su = su.values[t]
                                cur_e = e.values[t]
                                cur_a = a.values[t]
                                cur_r_a = r_a.values[t]
                                cur_p = p.values[t]
                                cur_sy = sy.values[t]
                                cur_c = c.values[t]
                                cur_r_c = r_c.values[t]
                                cur_h = h.values[t]
                                cur_r_h = r_h.values[t]
                                cur_i = i.values[t]
                                cur_r_i = r_i.values[t]
                                cur_r = r.values[t]
                                cur_d = d.values[t]
                                cur_cases = cases.values[t]
                                cur_pob = cur_su + cur_e + cur_a + cur_r_a + cur_p + cur_sy + cur_c + cur_r_c + cur_h +\
                                          cur_r_h + cur_i + cur_r_i + cur_r

                                # Run infection
                                percent = np.array(i_1) + np.array(i_2) if wv == 'M' else np.array(i_1)
                                percent_change = min(beta[cur_region] * np.dot(percent, contacts), 1.0)
                                contagion_sus = cur_su * percent_change
                                dsu_dt = {-contagion_sus
                                          }
                                de_dt = {contagion_sus,
                                         ((self.arrival_rate['ARRIVAL_RATE'][gv] * arrival_coefficient[cur_region]
                                           * cur_pob / dep_pob[gv]) if self.arrival_rate['START_DAY'][gv] <= t <=
                                                                       self.arrival_rate['END_DAY'][gv] else 0.0),
                                         -cur_e / t_e
                                         }
                                da_dt = {cur_e * (1 - p_s[ev]) / t_e,
                                         -cur_a / t_a
                                         }
                                dr_a_dt = {cur_a / t_a
                                           }
                                dp_dt = {cur_e * p_s[ev] / t_e,
                                         -cur_p / t_p
                                         }
                                dsy_dt = {cur_p / t_p,
                                          -cur_sy / t_sy
                                          }
                                dc_dt = {cur_sy * p_c[(ev, hv)] / t_sy,
                                         -cur_c / t_d[ev]
                                         }
                                dr_c_dt = {cur_c * (1 - p_c_d[(gv, ev, hv)]) / t_d[ev],
                                           -cur_r_c / (t_r[ev] - t_d[ev])
                                           }
                                dh_dt = {cur_sy * p_h[(ev, hv)] / t_sy,
                                         -cur_h / t_d[ev]
                                         }
                                dr_h_dt = {cur_h * (1 - p_h_d[(gv, ev, hv)]) / t_d[ev],
                                           -cur_r_h / (t_r[ev] - t_d[ev])
                                           }
                                di_dt = {cur_sy * p_i[(ev, hv)] / t_sy,
                                         -cur_i / t_d[ev]
                                         }
                                dr_i_dt = {cur_i * (1 - p_i_d[(gv, ev, hv)]) / t_d[ev],
                                           -cur_r_i / (t_r[ev] - t_d[ev])
                                           }
                                dr_dt = {cur_r_c / (t_r[ev] - t_d[ev]),
                                         cur_r_h / (t_r[ev] - t_d[ev]),
                                         cur_r_i / (t_r[ev] - t_d[ev])
                                         }
                                dd_dt = {cur_c * p_c_d[(gv, ev, hv)] / t_d[ev],
                                         cur_h * p_h_d[(gv, ev, hv)] / t_d[ev],
                                         cur_i * p_i_d[(gv, ev, hv)] / t_d[ev]
                                         }
                                dcases_dt = {cur_e * p_s[ev] / t_e
                                             }
                                v_c.values[t + 1] = 0.0
                                su.values[t + 1] = cur_su + float(sum(dsu_dt))
                                e.values[t + 1] = cur_e + float(sum(de_dt))
                                a.values[t + 1] = cur_a + float(sum(da_dt))
                                r_a.values[t + 1] = cur_r_a + float(sum(dr_a_dt))
                                p.values[t + 1] = cur_p + float(sum(dp_dt))
                                sy.values[t + 1] = cur_sy + float(sum(dsy_dt))
                                c.values[t + 1] = cur_c + float(sum(dc_dt))
                                r_c.values[t + 1] = cur_r_c + float(sum(dr_c_dt))
                                h.values[t + 1] = cur_h + float(sum(dh_dt))
                                r_h.values[t + 1] = cur_r_h + float(sum(dr_h_dt))
                                i.values[t + 1] = cur_i + float(sum(di_dt))
                                r_i.values[t + 1] = cur_r_i + float(sum(dr_i_dt))
                                r.values[t + 1] = cur_r + float(sum(dr_dt))
                                d.values[t + 1] = cur_d + float(sum(dd_dt))
                                cases.values[t + 1] = cur_cases + float(sum(dcases_dt))
                                h_c.values[t + 1] = cur_sy * p_h[(ev, hv)] / t_sy
                                c_t_c.values[t + 1] = h_c.values[t + 1] * \
                                                      self.attention_costs[(ev, hv, 'C')][type_params['cost']]
                                h_t_c.values[t + 1] = (h.values[t + 1] + r_h.values[t + 1]) * \
                                                      self.attention_costs[(ev, hv, 'H')][type_params['cost']]
                                i_t_c.values[t + 1] = (i.values[t + 1] + r_i.values[t + 1]) * \
                                                      self.attention_costs[(ev, hv, 'I')][type_params['cost']]
                                t_c.values[t + 1] = v_c.values[t + 1] + c_t_c.values[t + 1] + h_t_c.values[t + 1] + \
                                                    i_t_c.values[t + 1]
                                daly.values[t + 1] = sum([(c.values[t + 1] + r_c.values[t + 1]) * daly_vector['Home'],
                                                          (h.values[t + 1] + r_h.values[t + 1]) * daly_vector[
                                                              'Hospital'],
                                                          (i.values[t + 1] + r_i.values[t + 1]) * daly_vector['ICU'],
                                                          (r.values[t + 1]) * daly_vector['Recovered'],
                                                          (d.values[t + 1]) * daly_vector['Death']])
                # Demographics / Health degrees
                if run_type != 'calibration':
                    births = self.birth_rates[gv]*dep_pob[gv]
                    for state in alive_compartments:
                        previous_growing_o_s = births if state == 0 else 0.0
                        previous_growing_o_h = 0.0
                        previous_growing_m_s = 0.0
                        previous_growing_m_h = 0.0
                        for el in range(len(age_groups)):
                            cur_m_s = population[gv][age_groups[el]]['M']['S'][state].values[t+1]
                            cur_m_h = population[gv][age_groups[el]]['M']['H'][state].values[t+1]
                            cur_o_s = population[gv][age_groups[el]]['O']['S'][state].values[t + 1]
                            cur_o_h = population[gv][age_groups[el]]['O']['H'][state].values[t + 1]
                            growing_m_s = cur_m_s / 1826 if el < len(age_groups)-1 else 0.0
                            growing_m_h = cur_m_h / 1826 if el < len(age_groups)-1 else 0.0
                            growing_o_s = cur_o_s / 1826 if el < len(age_groups) - 1 else 0.0
                            growing_o_h = cur_o_h / 1826 if el < len(age_groups) - 1 else 0.0
                            dying_m_s = cur_m_s*self.death_rates[(gv, age_groups[el])] if state not in i_2_indexes \
                                else 0.0
                            dying_m_h = cur_m_h * self.death_rates[(gv, age_groups[el])] if state not in i_2_indexes \
                                else 0.0
                            dying_o_s = cur_o_s * self.death_rates[(gv, age_groups[el])] if state not in i_2_indexes \
                                else 0.0
                            dying_o_h = cur_o_h * self.death_rates[(gv, age_groups[el])] if state not in i_2_indexes \
                                else 0.0
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
                                       previous_growing_m_h * (1 - self.med_degrees[(gv, age_groups[el])]['M_TO_O']),
                                       previous_growing_o_h * (self.med_degrees[(gv, age_groups[el])]['O_TO_M']),
                                       previous_growing_o_s * (1-self.morbidity_frac[age_groups[el]]) *
                                       (self.med_degrees[(gv, age_groups[el])]['O_TO_M']),
                                        -growing_m_h,
                                        -dying_m_h}

                            do_h_dt = {previous_growing_o_s * self.morbidity_frac[age_groups[el]] *
                                       (1 - self.med_degrees[(gv, age_groups[el])]['O_TO_M']),
                                       previous_growing_m_h * (1 - self.med_degrees[(gv, age_groups[el])]['M_TO_O']),
                                       previous_growing_o_h * (self.med_degrees[(gv, age_groups[el])]['O_TO_M']),
                                       previous_growing_m_s * self.morbidity_frac[age_groups[el]] *
                                       self.med_degrees[(gv, age_groups[el])]['M_TO_O'],
                                       -growing_o_h,
                                       -dying_o_h}

                            population[gv][age_groups[el]]['M']['S'][state].values[t + 1] += sum(dm_s_dt)
                            population[gv][age_groups[el]]['M']['H'][state].values[t + 1] += sum(dm_h_dt)
                            population[gv][age_groups[el]]['O']['S'][state].values[t + 1] += sum(do_s_dt)
                            population[gv][age_groups[el]]['O']['H'][state].values[t + 1] += sum(do_h_dt)

                            previous_growing_o_s = growing_o_s
                            previous_growing_o_h = growing_o_h
                            previous_growing_m_s = growing_m_s
                            previous_growing_m_h = growing_m_h
        if run_type == 'calibration':
            results_array = np.zeros(shape=(sim_length + 1, 12))
            for gv in departments:
                cur_region = self.regions[gv] - 1
                for ev in age_groups:
                    for wv in work_groups:
                        for hv in health_groups:
                            results_array[:, cur_region] += np.array(list(population[gv][ev][wv][hv][14]
                                                                          .values.values()),
                                                            dtype=float)
                            results_array[:, cur_region+6] += np.array(list(population[gv][ev][wv][hv][13]
                                                                            .values.values()),
                                                            dtype=float)
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
                            for comp in population[gv][ev][wv][hv]:
                                pop_dict_h[comp.name] = comp.values
                            cur_pop_pandas = pd.DataFrame.from_dict(pop_dict_h).reset_index(drop=False).rename(
                                columns={'index': 'day'})
                            cur_pop_pandas['Health'] = hv
                            if len(pop_pandas_w) > 0:
                                pop_pandas_w = pd.concat([pop_pandas_w, cur_pop_pandas], ignore_index=True)
                            else:
                                pop_pandas_w = cur_pop_pandas.copy()
                            del cur_pop_pandas
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
