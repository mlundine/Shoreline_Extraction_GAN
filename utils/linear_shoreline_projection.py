import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import datetime
from scipy import stats
import os
plt.rcParams["figure.figsize"] = (16,6)

def plot_timeseries_with_fit(data, projection=10):
    folder = os.path.dirname(data)
    name = os.path.basename(data)
    name = os.path.splitext(name)[0]
    new_name = name+'_proj.png'
    fig_path = os.path.join(folder, new_name)

    df = pd.read_csv(data)
    df.reset_index()
    df = df.dropna()
    datetime_strings = df['datetime']
    shore_pos = df['distances']
    datetimes = [None]*len(datetime_strings)
    datetimes_seconds = [None]*len(datetimes)
    initial_time = datetime.datetime.strptime(datetime_strings[0], '%Y-%m-%d %H:%M:%S')
    for i in range(len(datetimes)):
        datetime_str = datetime_strings[i]
        t = datetime.datetime.strptime(datetime_str, '%Y-%m-%d %H:%M:%S')
        dt = t-initial_time
        dt_sec = dt.total_seconds()
        datetimes[i]=t
        datetimes_seconds[i] = dt_sec
    datetimes = np.array(datetimes)
    datetimes_seconds = np.array(datetimes_seconds)
    datetimes_years = datetimes_seconds/(60*60*24*365)

    x = datetimes_years
    y = shore_pos
    result1 = stats.linregress(x,y)
    slope1 = result1.slope
    intercept1 = result1.intercept
    r_value1 = result1.rvalue**2
    intercept_err = result1.intercept_stderr
    slope_err = result1.stderr
    lab = ('OLS\nSlope: ' +
          str(np.round(slope1,decimals=3)) + ' $+/-$ ' + str(np.round(slope_err, decimals=3)) +
          '\nIntercept: ' +
          str(np.round(intercept1,decimals=3)) + ' $+/-$ ' + str(np.round(intercept_err, decimals=3)) +
          '\n$R^2$: ' + str(np.round(r_value1,decimals=3)))
    fit1x = np.linspace(min(x),max(x)+projection,len(x))
    fit1y = slope1*fit1x + intercept1
    
    plt.plot(fit1x, fit1y, '--', color = 'red', label = lab)
    plt.plot(datetimes_years, shore_pos, label=name)
    plt.xlabel('Time (UTC)')
    plt.ylabel('Cross-Shore Position (m)')
    plt.xticks(range(0,38+projection, 5), range(1984, 2022+projection, 5))
    plt.xlim(min(x), max(x)+projection)
    plt.ylim(min(y), max(np.concatenate((y,fit1y))))
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig_path,dpi=300)
    plt.close()

    

