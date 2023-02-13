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
    new_name2 = name+'_residual.png'
    new_name3 = name+'_residual_hist.png'
    new_name4 = name+'_linear_trend.csv'
    fig_path = os.path.join(folder, new_name)
    fig_path2 = os.path.join(folder, new_name2)
    fig_path3 = os.path.join(folder, new_name3)
    csv_path1 = os.path.join(folder, new_name4)

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
    plt.minorticks_on()
    plt.tight_layout()
    plt.savefig(fig_path,dpi=300)
    plt.close()

    fit1x = x
    fit1y = slope1*fit1x + intercept1

    residual = shore_pos - fit1y
    plt.plot(datetimes, residual)
    plt.xlabel('Time (UTC)')
    plt.ylabel('Observed - Linear Trend (m)')
    plt.xlim(min(datetimes), max(datetimes))
    plt.ylim(min(residual), max(residual))
    plt.minorticks_on()
    plt.tight_layout()
    plt.savefig(fig_path2, dpi=300)
    plt.close()

    r_mean = np.mean(residual)
    r_std = np.std(residual)
    lab1 = 'n = ' + str(len(residual)) + '\n'
    lab2 = 'Mean = ' + str(np.round(r_mean, 3)) + '\n'
    lab3 = 'Std. Dev. = ' + str(np.round(r_std,3))
    lab=lab1+lab2+lab3
    plt.hist(residual, bins=30, label=lab)
    plt.xlabel('Linear Fit - Observed (m)')
    plt.ylabel('Count')
    plt.minorticks_on()
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig_path3, dpi=300)
    plt.close()

    new_df = {'time':datetimes,
              'linear_fit':fit1y,
              'residual':residual}
    new_df = pd.DataFrame(new_df)
    new_df.to_csv(csv_path1, index=False)
    new_df = None
    return slope1
    
              
