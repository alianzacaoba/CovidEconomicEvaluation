from typing import List, Any

from compartment import Compartment
import pandas as pd
import numpy as np
from tqdm import tqdm
import warnings
import json
from pyexcelerate import Workbook

warnings.simplefilter('error')


def calculate_vaccine_assignments(department_population: dict, day: int, vaccine_priority: list,
                                  vaccine_capacity: float, candidates_indexes: list):
    remaining_vaccines = vaccine_capacity
    assignation = dict()
    for group in vaccine_priority:
        ev, wv, hv = group
        assigned = 0.0
        if remaining_vaccines > 0.0:
            candidates = sum(department_population[ev][wv][hv][cv].values[day] for cv in candidates_indexes)
            if candidates > 0:
                assigned = min(remaining_vaccines, candidates)
                remaining_vaccines -= assigned
                assigned /= candidates
        assignation[(ev, wv, hv)] = assigned
        if assigned == 0:
            return assignation
    return assignation


class Model(object):
    age_groups: List[Any]
    departments: List[Any]
    work_groups: List[Any]

    def __init__(self):
        self.compartments = dict()
        initial_pop = pd.read_csv('input\\initial_population.csv', sep=';')
        self.departments = list(initial_pop.DEPARTMENT.unique())
        self.age_groups = list(initial_pop.AGE_GROUP.unique())
        self.work_groups = list(initial_pop.WORK_GROUP.unique())
        self.health_groups = list(initial_pop.HEALTH_GROUP.unique())
        with open('input\\neighbors.json') as json_file:
            self.neighbors = json.load(json_file)
        self.initial_population = dict()
        for g in self.departments:
            dep_department = dict()
            for e in self.age_groups:
                age_group = dict()
                for w in self.work_groups:
                    work_group = dict()
                    for h in self.health_groups:
                        work_group[h] = float(initial_pop[(initial_pop['DEPARTMENT'] == g) &
                                                          (initial_pop['AGE_GROUP'] == e) &
                                                          (initial_pop['WORK_GROUP'] == w) &
                                                          (initial_pop['HEALTH_GROUP'] == h)].POPULATION.sum())
                    age_group[w] = work_group
                dep_department[e] = age_group
            self.initial_population[g] = dep_department
        self.contact_matrix = dict()
        con_matrix = pd.read_csv('input\\contact_matrix.csv', sep=",")
        self.birth_rates = pd.read_csv('input\\birth_rate.csv', sep=';', index_col=0).to_dict()['BIRTH_RATE']
        morbidity_frac = pd.read_csv('input/morbidity_fraction.csv', sep=';', index_col=0)
        self.morbidity_frac = morbidity_frac.to_dict()['COMORBIDITY_RISK']
        del morbidity_frac
        self.death_rates = pd.read_csv('input\\death_rate.csv', sep=';', index_col=[0, 1]).to_dict()['DEATH_RATE']
        self.med_degrees = pd.read_csv('input\\medical_degrees.csv', sep=';', index_col=[0, 1]).to_dict(orient='index')
        for c in con_matrix.columns:
            self.contact_matrix[c] = con_matrix[c].to_list()
        self.arrival_rate = pd.read_csv('input\\arrival_rate.csv', sep=';', index_col=0).to_dict()

        time_params_load = pd.read_csv('input\\input_time.csv', sep=';')
        self.time_params = dict()
        for pv1 in time_params_load['PARAMETER'].unique():
            param = dict()
            current_param = time_params_load[time_params_load['PARAMETER'] == pv1]
            for ev in current_param.AGE_GROUP.unique():
                current_age_p = current_param[current_param['AGE_GROUP'] == ev]
                age = {'INF_VALUE': float(current_age_p['INF_VALUE'].sum()),
                       'BASE_VALUE': float(current_age_p['BASE_VALUE'].sum()),
                       'MAX_VALUE': float(current_age_p['MAX_VALUE'].sum())}
                del current_age_p
                param[ev] = age
                del age
            del current_param
            self.time_params[pv1] = param
        del time_params_load
        prob_params_load = pd.read_csv('input\\input_probabilities.csv', sep=';')
        self.prob_params = dict()
        for pv1 in prob_params_load['PARAMETER'].unique():
            param = dict()
            current_param = prob_params_load[prob_params_load['PARAMETER'] == pv1]
            for ev in current_param.AGE_GROUP.unique():
                age = dict()
                current_age_p = current_param[current_param['AGE_GROUP'] == ev]
                for hv in current_age_p['HEALTH_GROUP'].unique():
                    current_health = current_age_p[current_age_p['HEALTH_GROUP'] == hv]
                    health = {'INF_VALUE': float(current_health['INF_VALUE'].sum()),
                              'BASE_VALUE': float(current_health['BASE_VALUE'].sum()),
                              'MAX_VALUE': float(current_health['MAX_VALUE'].sum())}
                    age[hv] = health
                    del current_health
                param[ev] = age
                del current_age_p
                del age
            self.prob_params[pv1] = param
            del current_param

    def run(self, type_params: dict, name: str = 'Iteration', run_type: str = 'vaccination', beta: float = 0.5,
            death_coefficient: float = 1.0, vaccine_priority: list = None, vaccine_capacities: dict = None,
            vaccine_effectiveness: dict = None, vaccine_start_day: dict = None, vaccine_end_day: dict = None,
            calculated_arrival: bool = True, sim_length: int = 236, movement_coefficient: float = 0.01,
            vaccine_cost: float = 1.0, home_treatment_cost: float = 10.0, hospital_treatment_cost: float = 100.0,
            icu_treatment_cost: float = 1000.0, daly_vector: dict = None):
        # run_type:
        #   1) 'calibration': for calibration purposes, states f1,f2,v1,v2,e_f,a_f do not exist
        #   2) 'vaccination': model with vaccine states
        # SU, E, A, R_A, P, Sy, C, H, I, R, D, Cases
        population = dict()
        departments = self.departments
        age_groups = self.age_groups
        work_groups = self.work_groups
        health_groups = self.health_groups
        if daly_vector is None:
            daly_vector = {'Home': 0.2, 'Hospital': 0.3, 'ICU': 0.5, 'Death': 1, 'Recovered': 0.1}
    
        t_e = self.time_params['t_e']['ALL'][type_params['t_e']]
        t_p = self.time_params['t_p']['ALL'][type_params['t_p']]
        t_sy = self.time_params['t_sy']['ALL'][type_params['t_sy']]
        t_a = self.time_params['t_a']['ALL'][type_params['t_a']]
        initial_sus = self.prob_params['initial_sus']['ALL']['ALL'][type_params['initial_sus']]
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
            t_d[ev] = self.time_params['t_d'][ev][type_params['t_d']]
            t_r[ev] = self.time_params['t_r'][ev][type_params['t_r']]
            p_s[ev] = self.prob_params['p_s'][ev]['ALL'][type_params['p_s']]
            p_c[ev] = dict()
            p_h[ev] = dict()
            p_i[ev] = dict()
            p_c_d[ev] = dict()
            p_h_d[ev] = dict()
            p_i_d[ev] = dict()
            for hv in health_groups:
                p_c[ev][hv] = self.prob_params['p_c'][ev][hv][type_params['p_c']]
                p_h[ev][hv] = self.prob_params['p_h'][ev][hv][type_params['p_h']]
                p_i[ev][hv] = 1 - p_c[ev][hv] - p_h[ev][hv]
                p_c_d[ev][hv] = min(death_coefficient*self.prob_params['p_c_d'][ev][hv][type_params['p_c_d']], 1.0)
                p_h_d[ev][hv] = min(death_coefficient*self.prob_params['p_h_d'][ev][hv][type_params['p_h_d']], 1.0)
                p_i_d[ev][hv] = min(death_coefficient*self.prob_params['p_i_d'][ev][hv][type_params['p_i_d']], 1.0)

        arrival_rate = self.arrival_rate['CALCULATED_RATE'].copy() if calculated_arrival else \
            self.arrival_rate['SYMPTOMATIC_RATE'].copy()
        # 0) SU, 1) E, 2) A, 3) R_A, 4) P, 5) Sy, 6) C, 7) R_C, 8) H, 9) R_H, 10)I, 11) R_I 12) R, 13) D, 14)Cases,
        # 15) F1, 16) F2, 17) V1, 18) V2, 19) EF, 20) AF  15(NV)-21(V)) Home cases
        i_1_indexes = [2, 4, 5, 20] if run_type == 'vaccination' else [2, 4, 5]
        i_2_indexes = [6, 7, 8, 9, 10, 11]
        candidates_indexes = [0, 1, 2, 3, 15, 17] if run_type == 'vaccination' else [0, 1, 2, 3]
        alive_compartments = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 15, 16, 17, 18, 19, 20] \
            if run_type == 'vaccination' else [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]

        for gv in departments:
            population_g = dict()
            for ev in age_groups:
                population_e = dict()
                for wv in work_groups:
                    population_w = dict()
                    for hv in health_groups:
                        first_value = self.initial_population[gv][ev][wv][hv]
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
        for t in tqdm(range(sim_length)):
            dep_pob = dict()
            for gv in departments:
                if t > 50:
                    arrival_rate[gv] = 0.0
                if run_type == 'vaccination' and vaccine_start_day[gv] <= t <= vaccine_end_day[gv]:
                    vaccine_assignments = calculate_vaccine_assignments(department_population=population[gv],
                                                                             day=t, vaccine_priority=vaccine_priority,
                                                                             vaccine_capacity=vaccine_capacities[gv],
                                                                             candidates_indexes=candidates_indexes)
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
                    contacts = np.array(self.contact_matrix[ev])
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
                                    v_c.values[t + 1] = day_v.values[t + 1]*vaccine_cost
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
                                contagion_sus = cur_su * beta * (
                                            1 - float(np.prod(np.power(1 - percent, contacts))))
                                contagion_f_1 = cur_f_1 * beta * (
                                            1 - float(np.prod(np.power(1 - percent, contacts))))
                                contagion_f_2 = cur_f_2 * beta * (
                                            1 - float(np.prod(np.power(1 - percent, contacts))))
                                dsu_dt = {-contagion_sus}
                                df_1_dt = {-contagion_f_1}
                                df_2_dt = {-contagion_f_2}
                                de_dt = {contagion_sus,
                                         arrival_rate[gv] * cur_pob / dep_pob[gv],
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
                                dc_dt = {cur_sy * p_c[ev][hv] / t_sy,
                                         -cur_c / t_d[ev]
                                         }
                                dr_c_dt = {cur_c * (1 - p_c_d[ev][hv]) / t_d[ev],
                                           -cur_r_c / (t_r[ev] - t_d[ev])
                                           }
                                dh_dt = {cur_sy * p_h[ev][hv] / t_sy,
                                         -cur_h / t_d[ev]
                                         }
                                dr_h_dt = {cur_h * (1 - p_h_d[ev][hv]) / t_d[ev],
                                           -cur_r_h / (t_r[ev] - t_d[ev])
                                           }
                                di_dt = {cur_sy * p_i[ev][hv] / t_sy,
                                         -cur_i / t_d[ev]
                                         }
                                dr_i_dt = {cur_i * (1 - p_i_d[ev][hv]) / t_d[ev],
                                           -cur_r_i / (t_r[ev] - t_d[ev])
                                           }
                                dr_dt = {cur_r_c / (t_r[ev] - t_d[ev]),
                                         cur_r_h / (t_r[ev] - t_d[ev]),
                                         cur_r_i / (t_r[ev] - t_d[ev])
                                         }
                                dd_dt = {cur_c * p_c_d[ev][hv] / t_d[ev],
                                         cur_h * p_h_d[ev][hv] / t_d[ev],
                                         cur_i * p_i_d[ev][hv] / t_d[ev]
                                         }
                                dcases_dt = {cur_e * p_s[ev] / t_e}
                                dv_2_dt = {cur_a_f/t_a}

                                h_c.values[t + 1] = cur_sy * p_h[ev][hv] / t_sy
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

                                c_t_c.values[t + 1] = h_c.values[t + 1]*home_treatment_cost
                                h_t_c.values[t + 1] = (h.values[t + 1]+r_h.values[t + 1])*hospital_treatment_cost
                                i_t_c.values[t + 1] = (i.values[t + 1]+r_i.values[t + 1])*icu_treatment_cost
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
                                contagion_sus = cur_su * beta * (1 - float(np.prod(np.power(1-percent, contacts))))
                                dsu_dt = {-contagion_sus
                                          }
                                de_dt = {contagion_sus,
                                         arrival_rate[gv] * cur_pob / dep_pob[gv],
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
                                dc_dt = {cur_sy * p_c[ev][hv] / t_sy,
                                         -cur_c / t_d[ev]
                                         }
                                dr_c_dt = {cur_c * (1 - p_c_d[ev][hv]) / t_d[ev],
                                           -cur_r_c / (t_r[ev] - t_d[ev])
                                           }
                                dh_dt = {cur_sy * p_h[ev][hv] / t_sy,
                                         -cur_h / t_d[ev]
                                         }
                                dr_h_dt = {cur_h * (1 - p_h_d[ev][hv]) / t_d[ev],
                                           -cur_r_h / (t_r[ev] - t_d[ev])
                                           }
                                di_dt = {cur_sy * p_i[ev][hv] / t_sy,
                                         -cur_i / t_d[ev]
                                         }
                                dr_i_dt = {cur_i * (1 - p_i_d[ev][hv]) / t_d[ev],
                                           -cur_r_i / (t_r[ev] - t_d[ev])
                                           }
                                dr_dt = {cur_r_c / (t_r[ev] - t_d[ev]),
                                         cur_r_h / (t_r[ev] - t_d[ev]),
                                         cur_r_i / (t_r[ev] - t_d[ev])
                                         }
                                dd_dt = {cur_c * p_c_d[ev][hv] / t_d[ev],
                                         cur_h * p_h_d[ev][hv] / t_d[ev],
                                         cur_i * p_i_d[ev][hv] / t_d[ev]
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
                                h_c.values[t + 1] = cur_sy * p_h[ev][hv] / t_sy
                                c_t_c.values[t + 1] = h_c.values[t + 1] * home_treatment_cost
                                h_t_c.values[t + 1] = (h.values[t + 1] + r_h.values[t + 1]) * hospital_treatment_cost
                                i_t_c.values[t + 1] = (i.values[t + 1] + r_i.values[t + 1]) * icu_treatment_cost
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
            # Run mobility
            if t > 150 and run_type != "calibration":
                moving_pob = dict()
                for gv in departments:
                    for ev in age_groups:
                        for wv in work_groups:
                            for hv in health_groups:
                                for state in alive_compartments:
                                    if state not in i_2_indexes:
                                        moving_pob[(gv, ev, wv, hv, state)] = \
                                            population[gv][ev][wv][hv][state].values[t + 1]*movement_coefficient
                for gv in departments:
                    for ev in age_groups:
                        for wv in work_groups:
                            for hv in health_groups:
                                for state in alive_compartments:
                                    if state not in i_2_indexes:
                                        population[gv][ev][wv][hv][state].values[t + 1] -= \
                                            moving_pob[(gv, ev, wv, hv, state)]
                                        population[gv][ev][wv][hv][state].values[t + 1] += sum(
                                            moving_pob[(gv2, ev, wv, hv, state)] * dep_pob[gv] /
                                            sum(dep_pob[nv] for nv in self.neighbors[gv2])
                                            for gv2 in self.neighbors[gv])

        if run_type == 'calibration':
            results_array = np.zeros(shape=(sim_length + 1, 2))
            for gv in departments:
                for ev in age_groups:
                    for wv in work_groups:
                        for hv in health_groups:
                            results_array[:, 0] += np.array(list(population[gv][ev][wv][hv][14].values.values()),
                                                            dtype=float)
                            results_array[:, 1] += np.array(list(population[gv][ev][wv][hv][13].values.values()),
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
            with open('output\\result_' + name + '.json', 'w') as fp:
                json.dump(pop_dict, fp)
            print('JSon ', 'output\\result_' + name + '.json', 'exported')
            pop_pandas.to_csv('output\\result_' + name + '.csv', index=False)
            print('CSV ', 'output\\result_' + name + '.csv', 'exported')
            print('Begin excel exportation')
            wb = Workbook()
            for gv in tqdm(departments):
                pop_pandas_current = pop_pandas[pop_pandas['Department'] == gv].drop(columns='Department')
                values = [pop_pandas_current.columns] + list(pop_pandas_current.values)
                name_gv = gv if len(gv) < 31 else gv[:15]
                wb.new_sheet(name_gv, data=values)
            pop_pandas_current = pop_pandas.groupby('Department').sum().reset_index(drop=False)
            values = [pop_pandas_current.columns] + list(pop_pandas_current.values)
            print('Excel exportation country results')
            wb.new_sheet('Country_results', data=values)
            print('Saving excel results', 'output\\result_' + name + '.xlsx')
            wb.save('output\\result_' + name + '.xlsx')
            print('Excel ', 'output\\result_' + name + '.xlsx', 'exported')
            return pop_pandas
