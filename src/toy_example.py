import pandas as pd
from sklearn.metrics import mean_absolute_error

toy = pd.DataFrame({
    'Compound': ['SOFT', 'SOFT', 'HARD', 'HARD', 'SOFT', 'HARD'],
    'LapTimeSeconds': [90, 92, 95, 97, 91, 96]
})

medias = toy.groupby('Compound')['LapTimeSeconds'].mean()
print(medias)

toy['predicted'] = toy['Compound'].map(medias)
print(toy)

error = mean_absolute_error(toy['LapTimeSeconds'], toy['predicted'])
print(error)
