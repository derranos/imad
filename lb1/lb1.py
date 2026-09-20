import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv('./lb1/data.csv', sep=';')
print(df.head())
df['start_date_time'] = pd.to_datetime(df['start_date_time'])
print(df.shape)
print(df.describe())
std = df['RCORR_E'].std()
mean = df['RCORR_E'].mean()
mask = (abs(df['RCORR_E'] - mean) > 3 * std)
print((df[mask]).count())
lower = mean - 3 * std
upper = mean + 3 * std
sns.lineplot(data=df, x='start_date_time', y='RCORR_E')
plt.axhline(upper, color='red', linestyle='--')
plt.axhline(lower, color='red', linestyle='--')
plt.ylim(-3, 800)
plt.show()

d = 1 / 12960
print(f"{d:.6f}") # доля от всех данных
df_no_anomaly = df[~mask]
med = df_no_anomaly['RCORR_E'].median()
print(med)
df.loc[mask, 'RCORR_E'] = med
sns.lineplot(data=df, x='start_date_time', y='RCORR_E')
plt.show()