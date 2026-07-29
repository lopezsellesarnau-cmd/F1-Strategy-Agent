import fastf1
fastf1.Cache.enable_cache('data/cache')

session = fastf1.get_session(2023, 'Bahrain', 'R')
session.load()
print(session.laps.head())
