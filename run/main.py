from logic.main_run import MainRun


model_run = MainRun()
''' Current best results:
  beta : (0.0014072900227121035, 0.0012110689672789015, 0.0009942680804829176, 0.0006958321122570018, 0.0007593612195621051, 0.00033586229610363275)
  dc : (0.11015171321069671, 0.020838933766309586, 0.04698853873652189, 0.013753117225539173, 0.62537747549434, 0.027383032870303674)
  arrival : (15.878851338116327, 4.290547767179303, 16.083795918155936, 14.713731428337969, 20.919553709504676, 4.053605807740275)
  spc : 0.022718468344525077
  error_seroprevalence : (0.0256878786602386, 0.0058579891843715095, 0.003291447311646878, 0.015698849897622928, 0.18526394659279172, 0.042414150881458966)
  error_cases : (0.93566680851258, 0.9342594555680243, 0.9535913316709507, 1.0152630698849607, 0.9151631625459854, 0.6548941629044797)
  error_deaths : (0.9637849330741525, 0.9963843108273343, 0.9948650722295863, 0.997262907186347, 0.8510353941828452, 0.9429174093334524)
  error : 0.3705477354953117
'''

c_beta_base = (0.015979623745597444, 0.012110689672789041, 0.010139480614469267, 0.006958321122570045,
               0.007557581199354777, 0.003358622961036333)
c_death_base = (0.35432607625519363, 0.14154775696906677, 0.30680876047171135, 0.20951602575780792, 0.453552807780514,
                7.5)
c_arrival_base = (10.831876038710954, 4.290547767179307, 15.591861489326254, 14.713731428337997, 19.684588648640553,
                  4.053605807740285)
spc = 0.5
model_run.run_quality_test(c_beta_base=c_beta_base, c_death_base=c_death_base, c_arrival_base=c_arrival_base, spc=spc)
model_run.run(c_beta_base=c_beta_base, c_death_base=c_death_base, c_arrival_base=c_arrival_base, spc=spc)
model_run.run_immunity_loss_scenarios(c_beta_base=c_beta_base, c_death_base=c_death_base, c_arrival_base=c_arrival_base,
                                      spc=spc)
model_run.run_tornado(c_beta_base=c_beta_base, c_death_base=c_death_base, c_arrival_base=c_arrival_base, spc=spc)
model_run.run_sensibility_spc(c_beta_base=c_beta_base, c_death_base=c_death_base, c_arrival_base=c_arrival_base,
                              spc=spc)
model_run.run_sensibility_vac_eff(c_beta_base=c_beta_base, c_death_base=c_death_base, c_arrival_base=c_arrival_base,
                                  spc=spc)
model_run.run_sensibility_ifr(c_beta_base=c_beta_base, c_death_base=c_death_base, c_arrival_base=c_arrival_base,
                              spc=spc)
model_run.run_sensibility_vac_end(c_beta_base=c_beta_base, c_death_base=c_death_base, c_arrival_base=c_arrival_base,
                                  spc=spc)
model_run.run_sensibility_contact_variation(c_beta_base=c_beta_base, c_death_base=c_death_base,
                                            c_arrival_base=c_arrival_base, spc=spc)
