from model_run import Model
import pandas as pd

model_ex = Model()
type_paramsA = dict()
for pv in model_ex.time_params:
    type_paramsA[pv] = 'BASE_VALUE'
for pv in model_ex.prob_params:
    type_paramsA[pv] = 'BASE_VALUE'
vaccine_priority = [["e7", "M", "H"], ["e6", "M", "H"], ["e5", "M", "H"], ["e4", "M", "H"], ["e3", "M", "H"],
                    ["e2", "M", "H"], ["e1", "M", "H"], ["e0", "M", "H"], ["e7", "M", "S"], ["e6", "M", "S"],
                    ["e5", "M", "S"], ["e4", "M", "S"], ["e3", "M", "S"], ["e2", "M", "S"], ["e1", "M", "S"],
                    ["e0", "M", "S"], ["e7", "O", "H"], ["e6", "O", "H"], ["e5", "O", "H"], ["e4", "O", "H"],
                    ["e3", "O", "H"], ["e2", "O", "H"], ["e1", "O", "H"], ["e0", "O", "H"], ["e7", "O", "S"],
                    ["e6", "O", "S"], ["e5", "O", "S"], ["e4", "O", "S"], ["e3", "O", "S"], ["e2", "O", "S"],
                    ["e1", "O", "S"], ["e0", "O", "S"]]
vaccine_effectiveness = pd.read_csv('input\\vaccine_effectiveness_example.csv',sep=';',index_col=[0, 1])
vaccine_effectiveness = vaccine_effectiveness.to_dict()['VACCINE_EFFECTIVENESS']
vaccine_information = pd.read_csv('input\\region_capacities.csv', sep=';', index_col=[0]).to_dict()
model_ex.run(type_params=type_paramsA, name='vac_test', run_type='vaccination', beta=0.007832798540690137,
             sim_length=365*3, vaccine_priority=vaccine_priority, vaccine_effectiveness=vaccine_effectiveness,
             vaccine_capacities=vaccine_information['VACCINE_CAPACITY'],
             vaccine_start_day=vaccine_information['VACCINE_START_DAY'],
             vaccine_end_day=vaccine_information['VACCINE_END_DAY'])
