import fastf1
import pandas as pd
from sklearn.metrics import mean_absolute_error

fastf1.Cache.enable_cache('data/cache')


def load_race(year, gp):
    session = fastf1.get_session(year, gp, 'R')
    session.load()
    laps_clean = session.laps.pick_quicklaps().copy()
    laps_clean["LapTimeSeconds"] = laps_clean["LapTime"].dt.total_seconds()
    return laps_clean[["Compound", "LapTimeSeconds"]].dropna()


dataset_jeddah = load_race(2023, 'Jeddah')
dataset_bahrain = load_race(2023, 'Bahrain')
dataset_miami = load_race(2023, 'Miami')

dataset_train = pd.concat([dataset_jeddah, dataset_bahrain])
dataset_test = dataset_miami.copy()

medias = dataset_train.groupby("Compound")["LapTimeSeconds"].mean()

# Compuestos que aparecen en test pero no en train (p. ej. INTERMEDIATE/WET
# si Miami tuvo lluvia y las otras dos carreras no) se quedarían sin media
# y predicted = NaN — los quitamos antes de calcular el error.
dataset_test["predicted"] = dataset_test["Compound"].map(medias)
sin_prediccion = dataset_test["predicted"].isna().sum()
if sin_prediccion:
    print(f"{sin_prediccion} vueltas sin compuesto visto en train, se descartan")
dataset_test = dataset_test.dropna(subset=["predicted"])

error = mean_absolute_error(dataset_test["LapTimeSeconds"], dataset_test["predicted"])
print(f"MAE train=Jeddah+Bahrain, test=Miami: {error:.2f}s")
