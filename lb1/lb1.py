import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv('./lb1/data.csv', sep=';')
print(df.head())
df['start_date_time'] = pd.to_datetime(df['start_date_time'])
print(df.shape)
print(df.describe())
std = df['RCORR_E'].std()
print((df[abs(df['RCORR_E']) > 3 * std]).count())
sns.lineplot(data=df, x='start_date_time', y='RCORR_E')
plt.axhline(3 * std, color='red', linestyle='--')
plt.ylim(0, 800)
plt.show()

print(3681 / 12960) # доля от всех данных
df_no_anomaly = df[abs(df['RCORR_E']) <= 3 * std]
med = df_no_anomaly['RCORR_E'].median()
print(med)
mask = abs(df['RCORR_E']) > 3 * std
df.loc[mask, 'RCORR_E'] = med
sns.lineplot(data=df, x='start_date_time', y='RCORR_E')
plt.show()