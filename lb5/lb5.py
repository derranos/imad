import pandas as pd
import numpy as np
import pywt
import matplotlib.pyplot as plt


df = pd.read_csv('./lb5/data.csv', sep=';')
dates = pd.to_datetime(df['start_date_time'])
signal = df.iloc[:, 1].to_numpy(dtype=float)

pad_size = 5760
signal2 = np.pad(signal, (pad_size, pad_size), mode='symmetric')

wavelet = 'morl'
Fs = 0.016
fc = pywt.central_frequency(wavelet)
freqrange = np.array([0.000005, 0.016])
scalerange = fc / (freqrange * (1 / Fs))
scales = np.arange(scalerange[-1], scalerange[0] + 1e-9, 1)

coeffs, _ = pywt.cwt(
    signal2, scales, wavelet, sampling_period=1 / Fs, method='fft'
)
image = coeffs[:, pad_size:-pad_size]

ticks = np.append(np.arange(0, len(signal), 720), len(signal) - 1)
tick_labels = dates.iloc[ticks].dt.strftime('%d.%m %H:%M')

fig_signal, ax_signal = plt.subplots(figsize=(13, 5))
ax_signal.plot(dates, signal, color='steelblue', linewidth=0.8)
ax_signal.set_title('Исходный сигнал RCORR_E')
ax_signal.set_xlabel('Дата и время')
ax_signal.set_ylabel('Значение RCORR_E')
ax_signal.set_xticks(dates.iloc[ticks], tick_labels, rotation=30, ha='right')
ax_signal.grid(True, alpha=0.3)
fig_signal.tight_layout()

frequencies = pywt.scale2frequency(wavelet, scales) * Fs
target_frequencies = np.array([0.016, 0.008, 0.004, 0.002, 0.001, 0.0005, 0.0001, 0.000005])
frequency_ticks = np.array([np.argmin(np.abs(frequencies - f)) for f in target_frequencies])
frequency_labels = [f'{frequencies[i]:.1e}' for i in frequency_ticks]

fig_cwt, ax_cwt = plt.subplots(figsize=(13, 7))
limit = np.percentile(np.abs(image), 99)
plot = ax_cwt.imshow(
    np.abs(image),
    aspect='auto',
    origin='upper',
    extent=(0, len(signal), len(scales) - 0.5, -0.5),
    cmap='magma',
    vmin=0,
    vmax=limit,
)
ax_cwt.set_title(f'Вейвлет-преобразование ({wavelet})')
ax_cwt.set_xlabel('Дата и время')
ax_cwt.set_ylabel('Частота (Гц)')
ax_cwt.set_xticks(ticks, tick_labels, rotation=30, ha='right')
ax_cwt.set_yticks(frequency_ticks, frequency_labels)
fig_cwt.colorbar(plot, ax=ax_cwt, label='Амплитуда коэффициентов CWT')
fig_cwt.tight_layout()
plt.show()
