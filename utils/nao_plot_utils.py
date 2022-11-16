import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import datetime
plt.rcParams["figure.figsize"] = (16,12)

def plot_ts_with_nao(data_csv):

    save_name_png = os.path.splitext(data_csv)[0]+'_nao.png'
    root = os.getcwd()
    nao_file = os.path.join(root,r'nao_data.csv')
    nao_df = pd.read_csv(nao_file)
    nao_data = np.array(nao_df['nao'])
    nao_dates_str = nao_df['datetime']
    nao_dates = [None]*len(nao_dates_str)
    for i in range(len(nao_dates)):
        datetime_str = nao_dates_str[i]
        t = datetime.datetime.strptime(datetime_str, '%Y-%m-%d')
        nao_dates[i]=t
    nao_dates = np.array(nao_dates)


    pos_idx = np.where(nao_data >= 0)
    neg_idx = np.where(nao_data < 0)

    
    
    df = pd.read_csv(data_csv)
    df.reset_index()
    df = df.dropna()
    datetime_strings = df['datetime']
    datetimes_shore = [None]*len(datetime_strings)
    for i in range(len(datetimes_shore)):
        datetime_str = datetime_strings[i]
        t = datetime.datetime.strptime(datetime_str, '%Y-%m-%d %H:%M:%S')
        datetimes_shore[i]=t
    datetimes_shore = np.array(datetimes_shore)
    
    #make and save timeseries plot
    plt.subplot(2,1,1)
    plt.plot(datetimes_shore, df['distances'], color='k')
    plt.xlabel('Time (UTC)')
    plt.ylabel('Cross-Shore Position (m)')
    plt.xlim(min(datetimes_shore), max(datetimes_shore))
    plt.ylim(min(df['distances']), max(df['distances']))
    plt.minorticks_on()
    
    plt.subplot(2,1,2)
    plt.bar(nao_dates[pos_idx], nao_data[pos_idx], color='blue')
    plt.bar(nao_dates[neg_idx], nao_data[neg_idx], color='red')
    plt.ylabel('North Atlantic Oscillation')
    plt.xlim(min(datetimes_shore), max(datetimes_shore))
    plt.ylim(min(nao_data), max(nao_data))
    plt.minorticks_on()
    plt.tight_layout()
    plt.savefig(save_name_png, dpi=300)
    plt.close()
