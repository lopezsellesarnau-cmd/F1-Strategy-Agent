import fastf1
from sklearn.metrics import mean_absolute_error

fastf1.Cache.enable_cache('data/cache')
session = fastf1.get_session(2023, 'Bahrain', 'R')
session.load()

laps = session.laps
laps_clean = laps.pick_quicklaps()
laps_clean = laps_clean.copy()
laps_clean["LapTimeSeconds"] = laps_clean["LapTime"].dt.total_seconds()

dataset = laps_clean[["Compound", "LapTimeSeconds"]]
dataset = dataset.dropna()

medias = dataset.groupby("Compound")["LapTimeSeconds"].mean()
dataset["predicted"] = dataset["Compound"].map(medias)

error = mean_absolute_error(dataset["LapTimeSeconds"], dataset["predicted"])
print(error)
