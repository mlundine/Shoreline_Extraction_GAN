import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import datetime
from scipy import stats
import os
plt.rcParams["figure.figsize"] = (16,6)

def plot_timeseries_with_rolling_means_and_linear_fit(data, projection=10):
    folder = os.path.dirname(data)
    name = os.path.basename(data)
    name = os.path.splitext(name)[0]
    new_name = name+'roll_proj.png'
    new_name2 = name+'yearly_moving_avg.png'
    new_name3 = name+'sixmonth_moving_avg.png'
    new_name4 = name+'threemonth_moving_avg.png'
    fig_path = os.path.join(folder, new_name)
    fig_path2 = os.path.join(folder, new_name2)
    fig_path3 = os.path.join(folder, new_name3)
    fig_path4 = os.path.join(folder, new_name4)

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
    new_df = pd.DataFrame({'shoreline':list(y)},
                          index=list(datetimes))
    y1 = new_df.rolling('91D', min_periods=1).mean()
    y2 = new_df.rolling('182D',min_periods=1).mean()
    y3 = new_df.rolling('365D', min_periods=1).mean()


    
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
    #plt.plot(datetimes_years, shore_pos, label=name)
    plt.plot(datetimes_years, y1, color='violet', label='Three Month Moving Average')
    plt.plot(datetimes_years, y2, color='green', label='Six Month Moving Average')
    plt.plot(datetimes_years, y3, color='navy', label='Yearly Moving Average')
    plt.minorticks_on()
    plt.xlabel('Time (UTC)')
    plt.ylabel('Cross-Shore Position (m)')
    plt.xticks(range(0,38+projection, 5), range(1984, 2022+projection, 5))
    plt.xlim(min(x), max(x)+projection)
    plt.ylim(min(y), max(np.concatenate((y,fit1y))))
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig_path,dpi=300)
    plt.close()

    plt.plot(datetimes, y3.shoreline, color='navy', label='Yearly Moving Average')
    plt.minorticks_on()
    plt.xticks(rotation=90)
    plt.xlabel('Time (UTC)')
    plt.ylabel('Cross-Shore Position (m)')
    plt.xlim(min(datetimes), max(datetimes))
    plt.ylim(min(y3.shoreline), max(y3.shoreline))
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig_path2,dpi=300)
    plt.close()

    plt.plot(datetimes, y2.shoreline, color='navy', label='Six Month Moving Average')
    plt.minorticks_on()
    plt.xticks(rotation=90)
    plt.xlabel('Time (UTC)')
    plt.ylabel('Cross-Shore Position (m)')
    plt.xlim(min(datetimes), max(datetimes))
    plt.ylim(min(y2.shoreline), max(y2.shoreline))
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig_path3,dpi=300)
    plt.close()

    plt.plot(datetimes, y1.shoreline, color='navy', label='Three Month Moving Average')
    plt.minorticks_on()
    plt.xticks(rotation=90)
    plt.xlabel('Time (UTC)')
    plt.ylabel('Cross-Shore Position (m)')
    plt.xlim(min(datetimes), max(datetimes))
    plt.ylim(min(y1.shoreline), max(y1.shoreline))
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig_path4,dpi=300)
    plt.close()

