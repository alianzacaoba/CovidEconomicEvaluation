from compartment import Compartment
import pandas as pd
import numpy as np
from tqdm import tqdm
import warnings
import json
from pyexcelerate import Workbook

warnings.simplefilter('error')


class Model(object):
    def __init__(self):
        self.compartments = dict()
        initial_pop = pd.read_csv('input\\initial_population.csv', sep=';')
        self.departments = list(initial_pop.DEPARTMENT.unique())
        self.age_groups = list(initial_pop.AGE_GROUP.unique())
        self.work_groups = list(initial_pop.WORK_GROUP.unique())
        self.health_groups = list(initial_pop.HEALTH_GROUP.unique())
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
            death_coefficient: float = 1.0, calculated_arrival: bool = True, sim_length: int = 236):
        # run_type:
        #   1) 'calibration': for calibration purposes, states f1,f2,v1,v2,e_f,a_f do not exist
        #   2) 'vaccination': model with vaccine states
        # SU, E, A, R_A, P, Sy, C, H, I, R, D, Cases
        population = dict()
        departments = self.departments
        age_groups = self.age_groups
        work_groups = self.work_groups
        health_groups = self.health_groups
        t_e = self.time_params['t_e']['ALL'][type_params['t_e']]
        t_p = self.time_params['t_p']['ALL'][type_params['t_p']]
        t_sy = self.time_params['t_sy']['ALL'][type_params['t_sy']]
        t_r = self.time_params['t_r']['ALL'][type_params['t_r']]
        t_a = self.time_params['t_a']['ALL'][type_params['t_a']]
        initial_sus = self.prob_params['initial_sus']['ALL']['ALL'][type_params['initial_sus']]
        t_d = dict()
        p_s = dict()
        p_c = dict()
        p_h = dict()
        p_i = dict()
        p_c_d = dict()
        p_h_d = dict()
        p_i_d = dict()
        for ev in age_groups:
            t_d[ev] = self.time_params['t_d'][ev][type_params['t_d']]
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
                p_c_d[ev][hv] = death_coefficient*self.prob_params['p_c_d'][ev][hv][type_params['p_c_d']]
                p_h_d[ev][hv] = death_coefficient*self.prob_params['p_h_d'][ev][hv][type_params['p_h_d']]
                p_i_d[ev][hv] = death_coefficient*self.prob_params['p_i_d'][ev][hv][type_params['p_i_d']]

        arrival_rate = self.arrival_rate['CALCULATED_RATE'].copy() if calculated_arrival else \
            self.arrival_rate['SYMPTOMATIC_RATE'].copy()
        # 0) SU, 1) E, 2) A, 3) R_A, 4) P, 5) Sy, 6) C, 7) R_C, 8) H, 9) R_H, 10)I, 11) R_I 12) R, 13) D, 14)Cases,
        # 15) F1, 16) F2, 17) V1, 18) V2, 19) EF, 20) AF
        i_1_indexes = [2, 4, 5, 20] if run_type == 'vaccination' else [2, 4, 5]
        i_2_indexes = [6, 7, 8, 9, 10, 11]
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
                        population_w[hv] = compartments
                    population_e[wv] = population_w
                population_g[ev] = population_e
            population[gv] = population_g
        for t in tqdm(range(sim_length)):
            for gv in departments:
                if t > 50:
                    arrival_rate[gv] = 0.0
                i_1 = list()
                i_2 = list()
                dep_pob = 0.0
                age_pob = dict()
                for ev in age_groups:
                    tot = 0.0
                    inf1 = 0.0
                    inf2 = 0.0
                    for wv in work_groups:
                        for hv in health_groups:
                            tot += sum(comp.values[t] for comp in population[gv][ev][wv][hv])
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
                    dep_pob += tot
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
                                su, e, a, r_a, p, sy, c, r_c, h, r_h, i, r_i, r, d, cases, f_1, f_2, v_1, v_2, e_f, a_f \
                                    = population[gv][ev][wv][hv]
                                cur_f_1 = f_1.values[t]
                                cur_f_2 = f_2.values[t]
                                cur_v_1 = v_1.values[t]
                                cur_v_2 = v_2.values[t]
                                cur_e_f = e_f.values[t]
                                cur_a_f = a_f.values[t]
                            else:
                                su, e, a, r_a, p, sy, c, r_c, h, r_h, i, r_i, r, d, cases = population[gv][ev][wv][hv]
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

                            cur_pob = cur_su + cur_e + cur_a + cur_r_a + cur_p + cur_sy + cur_c + cur_r_c + cur_h + \
                                      cur_r_h + cur_i + cur_r_i + cur_r
                            # Run infection
                            percent = np.array(i_1) + np.array(i_2) if wv == 'M' else np.array(i_1)
                            contagion_sus = cur_su * beta * (1 - float(np.prod(np.power(1-percent, contacts))))
                            if run_type == 'vaccination':
                                contagion_f_1 = cur_f_1 * beta * (1 - float(np.prod(np.power(1 - percent, contacts))))
                                contagion_f_2 = cur_f_2 * beta * (1 - float(np.prod(np.power(1 - percent, contacts))))
                                dsu_dt = {-contagion_sus
                                          }
                                de_dt = {contagion_sus,
                                         arrival_rate[gv] * cur_pob / dep_pob,
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
                                           -cur_r_c / (t_r - t_d[ev])
                                           }
                                dh_dt = {cur_sy * p_h[ev][hv] / t_sy,
                                         -cur_h / t_d[ev]
                                         }
                                dr_h_dt = {cur_h * (1 - p_h_d[ev][hv]) / t_d[ev],
                                           -cur_r_h / (t_r - t_d[ev])
                                           }
                                di_dt = {cur_sy * p_i[ev][hv] / t_sy,
                                         -cur_i / t_d[ev]
                                         }
                                dr_i_dt = {cur_i * (1 - p_i_d[ev][hv]) / t_d[ev],
                                           -cur_r_i / (t_r - t_d[ev])
                                           }
                                dr_dt = {cur_r_c / (t_r - t_d[ev]),
                                         cur_r_h / (t_r - t_d[ev]),
                                         cur_r_i / (t_r - t_d[ev])
                                         }
                                dd_dt = {cur_c * p_c_d[ev][hv] / t_d[ev],
                                         cur_h * p_h_d[ev][hv] / t_d[ev],
                                         cur_i * p_i_d[ev][hv] / t_d[ev]
                                         }
                                dcases_dt = {cur_e * p_s[ev] / t_e
                                             }
                            else:
                                dsu_dt = {-contagion_sus
                                          }
                                de_dt = {contagion_sus,
                                         arrival_rate[gv] * cur_pob / dep_pob,
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
                                           -cur_r_c / (t_r - t_d[ev])
                                           }
                                dh_dt = {cur_sy * p_h[ev][hv] / t_sy,
                                         -cur_h / t_d[ev]
                                         }
                                dr_h_dt = {cur_h * (1 - p_h_d[ev][hv]) / t_d[ev],
                                           -cur_r_h / (t_r - t_d[ev])
                                           }
                                di_dt = {cur_sy * p_i[ev][hv] / t_sy,
                                         -cur_i / t_d[ev]
                                         }
                                dr_i_dt = {cur_i * (1 - p_i_d[ev][hv]) / t_d[ev],
                                           -cur_r_i / (t_r - t_d[ev])
                                           }
                                dr_dt = {cur_r_c / (t_r - t_d[ev]),
                                         cur_r_h / (t_r - t_d[ev]),
                                         cur_r_i / (t_r - t_d[ev])
                                         }
                                dd_dt = {cur_c * p_c_d[ev][hv] / t_d[ev],
                                         cur_h * p_h_d[ev][hv] / t_d[ev],
                                         cur_i * p_i_d[ev][hv] / t_d[ev]
                                         }
                                dcases_dt = {cur_e * p_s[ev] / t_e
                                             }
                            cur_su += float(sum(dsu_dt))
                            cur_e += float(sum(de_dt))
                            cur_a += float(sum(da_dt))
                            cur_r_a += float(sum(dr_a_dt))
                            cur_p += float(sum(dp_dt))
                            cur_sy += float(sum(dsy_dt))
                            cur_c += float(sum(dc_dt))
                            cur_r_c += float(sum(dr_c_dt))
                            cur_h += float(sum(dh_dt))
                            cur_r_h += float(sum(dr_h_dt))
                            cur_i += float(sum(di_dt))
                            cur_r_i += float(sum(dr_i_dt))
                            cur_r += float(sum(dr_dt))
                            cur_d += float(sum(dd_dt))
                            cur_cases += float(sum(dcases_dt))

                            # Run vaccine
                            su.values[t + 1] = cur_su
                            e.values[t + 1] = cur_e
                            a.values[t + 1] = cur_a
                            r_a.values[t + 1] = cur_r_a
                            p.values[t + 1] = cur_p
                            sy.values[t + 1] = cur_sy
                            c.values[t + 1] = cur_c
                            r_c.values[t + 1] = cur_r_c
                            h.values[t + 1] = cur_h
                            r_h.values[t + 1] = cur_r_h
                            i.values[t + 1] = cur_i
                            r_i.values[t + 1] = cur_r_i
                            r.values[t + 1] = cur_r
                            d.values[t + 1] = cur_d
                            cases.values[t + 1] = cur_cases
                # Run degrees
                # Run demographics
                # Run mobility

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
            values = [pop_pandas.columns] + list(pop_pandas.values)
            wb = Workbook()
            wb.new_sheet('All_values', data=values)
            wb.save('output\\result_' + name + '.xlsx')
            print('Excel ', 'output\\result_' + name + '.xlsx', 'exported')
            return pop_pandas


#model_ex = Model()
#type_paramsA = dict()
#for pv in model_ex.time_params:
#    type_paramsA[pv] = 'BASE_VALUE'
#for pv in model_ex.prob_params:
#    type_paramsA[pv] = 'BASE_VALUE'
#model_ex.run(type_params=type_paramsA, name='vac_test', run_type='vaccination', beta=0.007832798540690137)
