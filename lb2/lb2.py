import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import pairwise_distances

df = pd.read_csv('./lb2/data.csv', sep=';')
print(df.head())
df['start_date_time'] = pd.to_datetime(df['start_date_time'])
print(df.shape)
print(df.describe())
std = df['RCORR_E'].std()
mean = df['RCORR_E'].mean()
mask = (abs(df['RCORR_E'] - mean) > 3 * std)
print((df[mask]).count())

d = 1 / 12960
print(f"{d:.6f}") # доля от всех данных
df_no_anomaly = df[~mask]
med = df_no_anomaly['RCORR_E'].median()
print(med)
df.loc[mask, 'RCORR_E'] = med

m, k = 180, 4
data = df['RCORR_E']
vectors = np.lib.stride_tricks.sliding_window_view(data, m)
X_train = vectors[:-1]
X_test = vectors[-1:]
metrics = ['euclidean', 'manhattan', 'cosine']
for metric in metrics:
    dist = pairwise_distances(X_test, X_train, metric=metric).ravel()
    knn = dist.argsort()[:k]
    print(f"{metric}: расстояния = {dist[knn]}")
    avg = 0
    for ind in knn:
        avg += X_train[ind + m][0] / 4
    print(f"среднее предсказанное значение: {avg}")