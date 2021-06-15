from logic.main_run import MainRun


model_run = MainRun()

c_beta_base = (0.015979623745597444, 0.012110689672789041, 0.010139480614469267, 0.006958321122570045,
               0.007557581199354777, 0.003358622961036333)
c_death_base = (0.35432607625519363, 0.14154775696906677, 0.30680876047171135, 0.20951602575780792, 0.453552807780514,
                7.5)
c_arrival_base = (10.831876038710954, 4.290547767179307, 15.591861489326254, 14.713731428337997, 19.684588648640553,
                  4.053605807740285)
spc = 0.5
model_run.run_quality_test(c_beta_base=c_beta_base, c_death_base=c_death_base, c_arrival_base=c_arrival_base, spc=spc)
model_run.run(c_beta_base=c_beta_base, c_death_base=c_death_base, c_arrival_base=c_arrival_base, spc=spc)
model_run.run_tornado(c_beta_base=c_beta_base, c_death_base=c_death_base, c_arrival_base=c_arrival_base, spc=spc)
model_run.run_sensibility_vaccine_dist(c_beta_base=c_beta_base, c_death_base=c_death_base,
                                       c_arrival_base=c_arrival_base, spc=spc)
model_run.run_sensibility_contact_and_immunity_variation(c_beta_base=c_beta_base, c_death_base=c_death_base,
                                                         c_arrival_base=c_arrival_base, spc=spc)
model_run.run_sensibility_ifr_and_vac_end_variation(c_beta_base=c_beta_base, c_death_base=c_death_base,
                                                         c_arrival_base=c_arrival_base, spc=spc)
model_run.run_sensibility_vac_end(c_beta_base=c_beta_base, c_death_base=c_death_base, c_arrival_base=c_arrival_base,
                                  spc=spc)
model_run.run_sensibility_spc(c_beta_base=c_beta_base, c_death_base=c_death_base, c_arrival_base=c_arrival_base,
                              spc=spc)
model_run.run_immunity_loss_scenarios(c_beta_base=c_beta_base, c_death_base=c_death_base, c_arrival_base=c_arrival_base,
                                      spc=spc)
model_run.run_sensibility_ifr(c_beta_base=c_beta_base, c_death_base=c_death_base, c_arrival_base=c_arrival_base,
                              spc=spc)
model_run.run_sensibility_contact_variation(c_beta_base=c_beta_base, c_death_base=c_death_base,
                                            c_arrival_base=c_arrival_base, spc=spc)
model_run.run_sensibility_vac_eff(c_beta_base=c_beta_base, c_death_base=c_death_base, c_arrival_base=c_arrival_base,
                                  spc=spc)

