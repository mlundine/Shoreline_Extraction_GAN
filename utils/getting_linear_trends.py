# load modules
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import geopandas as gpd
import datetime
import shapely
from math import degrees, atan2, radians
from scipy import stats
plt.rcParams["figure.figsize"] = (16,6)

def plot_timeseries_with_fit(data,
                             min_month_filter = 12,
                             min_day_filter = 31,
                             min_year_filter = 1984,
                             max_month_filter = 1,
                             max_day_filter = 1,
                             max_year_filter = 2023):
    folder = os.path.dirname(data)
    lt_folder = os.path.join(folder, 'linear_trends'+str(min_year_filter)+'_'+str(max_year_filter))
    try:
        os.mkdir(lt_folder)
    except:
        pass
    name = os.path.basename(data)
    name = os.path.splitext(name)[0]
    new_name = name+'linear_trend.png'
    new_name_csv = name+'yearlyrunningmean.csv'
    fig_path = os.path.join(lt_folder, new_name)
    csv_path = os.path.join(lt_folder, new_name_csv)
    df = pd.read_csv(data)
    df.reset_index()
    df = df.dropna()
    datetime_strings = df['datetime']
    datetimes = [None]*len(datetime_strings)
    for i in range(len(datetimes)):
        datetime_str = datetime_strings[i]
        t = datetime.datetime.strptime(datetime_str, '%Y-%m-%d %H:%M:%S')
        datetimes[i] = t
    
    df['datetimes'] = datetimes
    filter_df = df[df['datetimes']>datetime.datetime(year=min_year_filter-1,month=min_month_filter,day=min_day_filter)]
    filter_df.reset_index()
    filter_df = filter_df[filter_df['datetimes']<datetime.datetime(year=max_year_filter+1,month=max_month_filter,day=max_day_filter)]
    filter_df.reset_index()

    filtered_datetimes = np.array(filter_df['datetimes'])
    shore_pos = np.array(filter_df['distances'])
    datetimes_seconds = [None]*len(filtered_datetimes)
    initial_time = filtered_datetimes[0]
    for i in range(len(filter_df)):
        t = filter_df['datetimes'].iloc[i]
        dt = t-initial_time
        dt_sec = dt.total_seconds()
        datetimes_seconds[i] = dt_sec
    datetimes_seconds = np.array(datetimes_seconds)
    datetimes_years = datetimes_seconds/(60*60*24*365)


    x = datetimes_years
    y = shore_pos
    
    new_df = pd.DataFrame({'shoreline':list(y)},
                          index=list(filtered_datetimes))
    y2 = new_df.rolling('365D', min_periods=1).mean()
    y2 = np.array(y2.shoreline)
    result1 = stats.linregress(x,y2)
    intercept1 = result1.intercept
    slope1 = result1.slope
    r_value1 = result1.rvalue**2
    intercept_err = result1.intercept_stderr
    slope_err = result1.stderr
    lab = ('OLS\nSlope: ' +
          str(np.round(slope1,decimals=3)) + ' $+/-$ ' + str(np.round(slope_err, decimals=3)) +
          '\nIntercept: ' +
          str(np.round(intercept1,decimals=3)) + ' $+/-$ ' + str(np.round(intercept_err, decimals=3)) +
          '\n$R^2$: ' + str(np.round(r_value1,decimals=3)))
    fit1x = datetimes_years
    fit1y = slope1*fit1x + intercept1
    
    plt.plot(filtered_datetimes, y2, color='navy', label='Yearly Moving Average')
    plt.plot(filtered_datetimes, fit1y, '--', color='red', label=lab)
    plt.minorticks_on()
    plt.xticks(rotation=90)
    plt.xlabel('Time (UTC)')
    plt.ylabel('Cross-Shore Position (m)')
    plt.xlim(min(filtered_datetimes), max(filtered_datetimes))
    plt.ylim(min(y2), max(y2))
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig_path,dpi=300)
    plt.close()

    new_df.to_csv(csv_path)
    
    return slope1, fig_path, csv_path

def gb(x1, y1, x2, y2):
    angle = degrees(atan2(y2 - y1, x2 - x1))
    bearing = angle
    #bearing = (90 - angle) % 360
    return bearing

def arr_to_LineString(coords):
    """
    Makes a line feature from a list of xy tuples
    inputs: coords
    outputs: line
    """
    points = [None]*len(coords)
    i=0
    for xy in coords:
        points[i] = shapely.geometry.Point(xy)
        i=i+1
    line = shapely.geometry.LineString(points)
    return line

def batch_transect_timeseries(sitename,
                              transects_csv_folder,
                              transects_path,
                              epsg,
                              min_month_filter = 12,
                              min_day_filter = 31,
                              min_year_filter = 1984,
                              max_month_filter = 1,
                              max_day_filter = 1,
                              max_year_filter = 2023):
    """
    Makes new transect shapefile with linear trends
    """
    transects_path_name = os.path.splitext(transects_path)[0]+'_trends'+str(min_year_filter)+'to'+str(max_year_filter)+'.shp'
    transects = gpd.read_file(transects_path)
    transects = transects.reset_index()
    slopes = [None]*len(transects)
    fig_paths = [None]*len(transects)
    csvs = [None]*len(transects)
    csv_paths = [None]*len(transects)
    for i in range(len(transects)):
        print(np.round(i/len(transects),2))
        csv = os.path.join(transects_csv_folder, sitename+str(i)+'.csv')
        slope, fig_path, csv_path = plot_timeseries_with_fit(csv,
                                                             min_month_filter = min_month_filter,
                                                             min_day_filter = min_day_filter,
                                                             min_year_filter = min_year_filter,
                                                             max_month_filter = max_month_filter,
                                                             max_day_filter = max_day_filter,
                                                             max_year_filter = max_year_filter)
        slopes[i] = slope
        csvs[i] = csv
        fig_paths[i] = fig_path
        csv_paths[i] = csv_path

    transects['raw_csv'] = csvs
    transects['yearly_trend'] = slopes
    transects['fig_paths'] = fig_paths
    transects['new_csv_paths'] = csv_paths
    max_slope = np.max(np.abs(slopes))
    scaled_slopes = (np.array(slopes)/1)*100
    new_lines = [None]*len(transects)
    for i in range(len(transects)):
        first, last = transects['geometry'][i].boundary
        midpoint = transects['geometry'][i].centroid
        distance = scaled_slopes[i]
        if distance<0:
            angle = radians(gb(first.x, first.y, last.x, last.y)+180)
        else:
            angle = radians(gb(first.x, first.y, last.x, last.y))
        northing = midpoint.y + abs(distance)*np.sin(angle)
        easting = midpoint.x + abs(distance)*np.cos(angle)
        line_arr = [(midpoint.x,midpoint.y),(easting,northing)]
        line = arr_to_LineString(line_arr)
        new_lines[i] = line
    
    new_df = pd.DataFrame({'year_tr':slopes,
                           'raw_csv':csvs,
                           'fig_paths':fig_paths,
                           'csv_paths':csv_paths})
    new_geo_df = gpd.GeoDataFrame(new_df, crs="EPSG:"+str(epsg), geometry=new_lines)
    
    new_geo_df.to_file(transects_path_name)




