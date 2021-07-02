import pandas as pd
import datetime as dt
from root import DIR_INPUT

print('Importing data', dt.datetime.now())
df = pd.read_csv('https://www.gstatic.com/covid19/mobility/Global_Mobility_Report.csv')
print('Information downloaded', dt.datetime.now())
df = df[df['country_region'] == 'Colombia']
print('Colombia information selected', dt.datetime.now())
reg_numbers = {'Amazonas': 6, 'Antioquia': 2, 'Arauca': 6, 'San Andres and Providencia': 1, 'Atlantico': 1, 'Bogota': 3,
               'Bolivar': 1,  'Boyaca': 4,  'Caldas': 2,  'Caqueta': 2, 'Casanare': 6, 'Cauca': 5, 'Cesar': 1,
               'Choco': 5, 'Cordoba': 1, 'Cundinamarca': 3, 'Guaviare': 6, 'Huila': 2, 'La Guajira': 1, 'Magdalena': 1,
               'Meta': 4, 'Narino': 5, 'North Santander': 4, 'Putumayo': 6, 'Quindio': 2, 'Risaralda': 2,
               'Santander': 4, 'Sucre': 1, 'Tolima': 2, 'Valle del Cauca': 5, 'Vichada': 6}
df['DATE'] = pd.to_datetime(df.date, yearfirst=True, errors='coerce')
df['REGIONS'] = df.sub_region_1.apply(lambda x: reg_numbers.get(x, 'N/A'))
df = df[['REGIONS', 'DATE', 'retail_and_recreation_percent_change_from_baseline',
         'grocery_and_pharmacy_percent_change_from_baseline',
         'parks_percent_change_from_baseline', 'transit_stations_percent_change_from_baseline',
         'workplaces_percent_change_from_baseline',
         'residential_percent_change_from_baseline']][df.REGIONS != 'N/A']
df = df.groupby(['REGIONS', 'DATE']).mean().reset_index(drop=False)
df['OTHER'] = df[['retail_and_recreation_percent_change_from_baseline',
                  'grocery_and_pharmacy_percent_change_from_baseline',
                  'parks_percent_change_from_baseline',
                  'transit_stations_percent_change_from_baseline']].mean(axis=1)
df.rename(columns={'workplaces_percent_change_from_baseline': 'WORK',
                   'residential_percent_change_from_baseline': 'HOME'}, inplace=True)
df = df[['REGIONS', 'DATE', 'WORK', 'HOME', 'OTHER']]
df = df[df.DATE >= dt.datetime(year=2020, month=2, day=21)]
df['SIM_DAY'] = (df.DATE - dt.datetime(year=2020, month=2, day=21)).dt.days
df[['WORK', 'HOME', 'OTHER']] = df[['WORK', 'HOME', 'OTHER']]/100
df = df[['DATE', 'SIM_DAY', 'REGIONS', 'WORK', 'HOME', 'OTHER']]
df['SCHOOL'] = 0
df.fillna(df.mean(), inplace=True)
print('Values selected', dt.datetime.now())
df.to_csv(DIR_INPUT + 'contact_matrix_coefficients.csv', index=False)
print('Update Finished', dt.datetime.now())
