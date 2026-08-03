import fastf1
import pandas as pd

fastf1.Cache.enable_cache('data/cache')

session_jeddah = fastf1.get_session(2023, 'Jeddah', 'R')
session_jeddah.load()

laps_jeddah = session_jeddah.laps
laps_jeddah_clean = laps_jeddah.pick_quicklaps().copy()
laps_jeddah_clean["LapTimeSeconds"] = laps_jeddah_clean["LapTime"].dt.total_seconds()
dataset_jeddah = laps_jeddah_clean[['Compound', 'LapTimeSeconds']].dropna()

session_bahrain = fastf1.get_session(2023, "Bahrain", 'R')
session_bahrain.load()

laps_bahrain = session_bahrain.laps
laps_bahrain_clean = laps_bahrain.pick_quicklaps().copy()
laps_bahrain_clean["LapTimeSeconds"] = laps_bahrain_clean["LapTime"].dt.total_seconds()
dataset_bahrain = laps_bahrain_clean[['Compound', 'LapTimeSeconds']].dropna()

session_miami = fastf1.get_session(2023, 'Miami', 'R')
session_miami.load()

laps_miami = session_miami.laps
laps_miami_clean = laps_miami.pick_quicklaps().copy()
laps_miami_clean["LapTimeSeconds"] = laps_miami_clean["LapTime"].dt.total_seconds()
dataset_miami = laps_miami_clean[['Compound', 'LapTimeSeconds']].dropna()

dataset_train = pd.concat([dataset_jeddah, dataset_bahrain,])
print(dataset_train.shape)

print(dataset_miami.shape)
