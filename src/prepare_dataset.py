import fastf1

fastf1.Cache.enable_cache('data/cache')
session = fastf1.get_session(2023, 'Bahrain', 'R')
session.load()

laps = session.laps
laps_clean = laps.pick_quicklaps().copy()
laps_clean['LapTimeSeconds'] = laps_clean['LapTime'].dt.total_seconds()

# Features + target en un solo DataFrame limpio
dataset = laps_clean[['Driver', 'LapNumber', 'TyreLife',
                      'Compound', 'Stint', 'LapTimeSeconds']].copy()
dataset = dataset.dropna()  # quita filas con datos faltantes

print(dataset.shape)
print(dataset.head())
print(dataset['Compound'].value_counts())
