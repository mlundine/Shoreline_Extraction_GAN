import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
import warnings
warnings.filterwarnings("ignore")
plt.rcParams["figure.figsize"] = (10,8)



def DFT(x, dt):
    x = x
    N = len(x) #gets record length of x
    T_0 = N*dt # gets actual time length of data
    dft = [None]*int(N/2) # list to fill with dft values
    freqs = [None]*int(N/2) # list to fill with frequencies
    ## Outer loop over frequencies, range(start, stop) goes from start to stop-1
    for n in range(0, int(N/2)):
        dft_element = 0
        # inner loop over all data
        for k in range(0,N):
            m = np.exp((-2j * np.pi * k * n)/N)
            dft_element = dft_element + x[k]*m
        dft_element = dft_element * dt
        freq_element = n/T_0
        dft[n] = dft_element
        freqs[n] = freq_element
    ## change dft from list to np array
    dft = np.array(dft)
    ## compute power spectrum
    ps = 2*np.abs(dft**2)/T_0
    freqs = np.array(freqs)
    return freqs, ps

## Plots the frequency vs the spectral density
def plot_power_spectrum(ps, freqs, save_path):
    """
    Plots power spectrum, y on log scale
    inputs:
    ps: power spectrum
    freqs: frequency bins
    """
    plt.plot(freqs, np.log(ps))
    freq_range = np.linspace(min(freqs), max(freqs), num=10)
    freq_labels = np.round(1/freq_range, 1)
    plt.xticks(freq_range, freq_labels)
    plt.xlabel('Period (Months)')
    lab = r'$ln(\frac{m^2}{cycles/s})$'
    plt.ylabel(r'Spectral Density ('+lab+')')
    plt.xlim(min(freqs), max(freqs))
    plt.minorticks_on()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

def get_shoreline_data(csv_path, resample_period = '45D'):
    ##importing data
    df = pd.read_csv(csv_path)
    new_df = pd.DataFrame({'datetime':pd.to_datetime(df['datetime']),
                           'value':df['distances']})
    new_df = new_df.drop_duplicates('datetime', 'last')
    new_df = new_df.set_index(['datetime'])
    new_df = new_df.dropna()
    y_rolling = new_df.rolling(resample_period, min_periods=1).mean()
    y1 = y_rolling.resample(resample_period).ffill()
    y1 = y1.dropna()
    df=y1
    df['Date'] = pd.to_datetime(df.index)
    df.set_axis(df['Date'], inplace=True)
    return df

def main(csv_path, resample_period='45D'):
    new_df = get_shoreline_data(csv_path, resample_period=resample_period)
    save_path = os.path.splitext(csv_path)[0]+'_dft.png'
    freqs, ps = DFT(new_df['value'], 1.5)
    plot_power_spectrum(ps, freqs, save_path)

