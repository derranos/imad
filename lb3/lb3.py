import pandas as pd
import numpy as np
import skfuzzy as fuzz
from sklearn.preprocessing import StandardScaler

def fill_med(data):
    data['start_date_time'] = pd.to_datetime(data['start_date_time'])
    data = data.set_index('start_date_time').sort_index()
    data = data[~data.index.duplicated()]

    full_range = pd.DatetimeIndex([])
    for day in data.index.normalize().unique():
        full_range = full_range.append(pd.date_range(day, periods=1440, freq='1min'))

    data = data.reindex(full_range)
    data['RCORR_E'] = data.groupby(data.index.date)['RCORR_E'].transform(
        lambda x: x.fillna(x.median())
    )
    return data

def feature_engineering(data):
    features = data.groupby(data.index.date)['RCORR_E'].agg(['mean', 'var', 'min', 'max',
                                                            ('q80', lambda x: x.quantile(0.8))])
    features['amplitude'] = features['max'] - features['min']
    return features[['mean', 'var', 'amplitude', 'q80']].rename_axis('date').reset_index()

data_calm = pd.read_csv('./calm.csv', sep=';')
data_storm = pd.read_csv('./storm.csv', sep=';')

data_calm = fill_med(data_calm)
data_storm = fill_med(data_storm)

features_calm = feature_engineering(data_calm)
features_storm = feature_engineering(data_storm)

features_calm['class'] = 1
features_storm['class'] = 2

data = pd.concat([features_calm, features_storm], ignore_index=True)

if data[['mean', 'var', 'amplitude', 'q80']].isna().any().any():
    raise ValueError('Не хватает данных для некоторых дат')

print(data.head())
print(data.shape)

X = data[['mean', 'var', 'amplitude', 'q80']].to_numpy()

scaler = StandardScaler()
X = scaler.fit_transform(X)

centers, u, _, _, _, _, fpc = fuzz.cluster.cmeans(
    X.T, c=2, m=2, error=0.005, maxiter=1000, seed=42
)

clusters = u.argmax(axis=0)
real = data['class'].to_numpy() - 1

if np.mean(clusters == real) < np.mean(1 - clusters == real):
    clusters = 1 - clusters

data['cluster'] = clusters + 1

print('Центры кластеров:')
print(scaler.inverse_transform(centers))

print('Коэффициент разбиения:', fpc)
print(pd.crosstab(data['class'], data['cluster']))
print('Доля совпадений:', np.mean(data['class'] == data['cluster']))

data.to_csv('./clusters.csv', sep=';', index=False)