import fastf1
import matplotlib.pyplot as plt

fastf1.Cache.enable_cache('data/cache')
session = fastf1.get_session(2023, 'Bahrain', 'R')
session.load()

laps = session.laps
laps_clean = laps.pick_quicklaps()
laps_clean = laps_clean.copy()
laps_clean['LapTimeSeconds'] = laps_clean['LapTime'].dt.total_seconds()

for compound in laps_clean['Compound'].unique():
    subset = laps_clean[laps_clean['Compound'] == compound]
    plt.scatter(subset['TyreLife'], subset['LapTimeSeconds'],
                label=compound, alpha=0.5)

plt.xlabel('Tyre Life (laps)')
plt.ylabel('Lap Time (s)')
plt.legend()
plt.savefig('data/degradation.png')
print("saved plot")
