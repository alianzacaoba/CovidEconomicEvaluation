from model_run import Model

model_ex = Model()
type_paramsA = dict()
for pv in model_ex.time_params:
    type_paramsA[pv] = 'BASE_VALUE'
for pv in model_ex.prob_params:
    type_paramsA[pv] = 'BASE_VALUE'
vaccine_priority = []
model_ex.run(type_params=type_paramsA, name='vac_test', run_type='vaccination', beta=0.007832798540690137,
             sim_length=365*3, vaccine_priority = vaccine_priority)

