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

def plot_timeseries_with_fit(data, min_year_filter = 1984, max_year_filter = 2023):

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
    filter_df = df[df['datetimes']>datetime.datetime(min_year_filter-1,12,31)]
    filter_df = df[df['datetimes']<datetime.datetime(max_year_filter+1,1,1)]
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
    result1 = stats.linregress(x,y)
    slope1 = result1.slope

    return slope1

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
                              min_year_filter=1984,
                              max_year_filter=2023):
    """
    Makes new transect shapefile with linear trends
    """
    transects_path_name = os.path.splitext(transects_path)[0]+'_trends'+str(min_year_filter)+'to'+str(max_year_filter)+'.shp'
    transects = gpd.read_file(transects_path)
    transects = transects.reset_index()
    slopes = [None]*len(transects)
    csvs = [None]*len(transects)
    for i in range(len(transects)):
        print(np.round(i/len(transects),2))
        csv = os.path.join(transects_csv_folder, sitename+str(i)+'.csv')
        slope = plot_timeseries_with_fit(csv, min_year_filter=min_year_filter,max_year_filter=max_year_filter)
        slopes[i] = slope

        
    transects['yearly_trend'] = slopes
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
    
    new_df = pd.DataFrame({'year_tr':slopes})
    new_geo_df = gpd.GeoDataFrame(new_df, crs="EPSG:"+str(epsg), geometry=new_lines)
    
    new_geo_df.to_file(transects_path_name)

