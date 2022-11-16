# load modules
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import geopandas as gpd
import datetime
from math import degrees, atan2
from utils import linear_shoreline_projection as lsp
from utils import rolling_mean as rm
from utils import timeseries_analysis_utils as tsa
from utils import nao_plot_utils as nao
plt.rcParams["figure.figsize"] = (16,6)

def gb(x1, y1, x2, y2):
    angle = degrees(atan2(y2 - y1, x2 - x1))
    bearing = (90 - angle) % 360
    return bearing

def transect_timeseries(shoreline_shapefile,
                        transect_shapefile,
                        sitename,
                        transect_id,
                        output_folder,
                        switch_dir=False,
                        batch=False):
    """
    Generates timeseries of shoreline cross-shore position
    given a shapefile containing shorelines and a shapefile containing
    a cross-shore transect. Computes interesection points between shorelines
    and transect. Uses earliest shoreline intersection point as origin.
    
    inputs:
    shoreline_shapefile (str): path to shapefile containing shorelines
                               (needs a field 'datetime' YYYY-MM-DD-HH-MM)
    transect_shapefile (str): path to shapefile containing cross-shore transect
    sitename (str): name of site
    transect_id (int): integer id for transect
    output_folder (str): path to save csv and png figure to
    switch_dir (optional): default is False, set to True if transects are in the opposite direction
    batch (optional): default is False, this gets set to True when batch function is used
    """
    #load shorelines
    shoreline_df = gpd.read_file(shoreline_shapefile)
    shoreline_df = shoreline_df.reset_index()

    #load transect
    if batch == False:
        transect_df = gpd.read_file(transect_shapefile)
    else:
        transect_df = transect_shapefile
    first, last = transect_df['geometry'][0].boundary
    bearing = gb(first.x, first.y, last.x, last.y)
    if switch_dir == True:
        bearing = bearing-180
    #get intersection points
    northings = [None]*len(shoreline_df)
    eastings = [None]*len(shoreline_df)
    dates = [None]*len(shoreline_df)
    # loop through shorelines and compute the intersection    
    for i in range(len(shoreline_df)):
        sl = shoreline_df[shoreline_df['index']==i]
        try:
            point = sl.unary_union.intersection(transect_df.unary_union)
            easting = point.x
            northing = point.y
        except:
            continue

        
        northings[i] = northing
        eastings[i] = easting
        date = sl['timestamp'].reset_index().values[0][1]
        dates[i] = datetime.datetime(*map(int, date.split('-'))) 
    df_dict = {'datetime':dates,
               'northings':northings,
               'eastings':eastings}
    df = pd.DataFrame(df_dict)
    df = df.sort_values(by=['datetime'], ascending=True)
    df = df.reset_index()

    # get cross-shore distance
    distances = [None]*len(df)
    distances[0] = 0
    x1 = df['eastings'][0]
    y1 = df['northings'][0]
    for i in range(1, len(df)):
        x_0 = df['eastings'][0]
        y_0 = df['northings'][0]
        x_n = df['eastings'][i]
        y_n = df['northings'][i]
        distance = np.sqrt((x_n-x_0)**2+(y_n-y_0)**2)
        new_bearing = gb(x_0, y_0, x_n, y_n)
        if abs(bearing - new_bearing) < 0.00001:
            distances[i] = distance
        else:
            distances[i] = -distance
    df['distances'] = distances
    df = df.reset_index(drop=True)
    
    
    #paths to save
    save_name_png = os.path.join(output_folder, sitename+'_'+str(transect_id)+'.png')
    save_name_csv = os.path.join(output_folder, sitename+'_'+str(transect_id)+'.csv')

    #save csv
    df.to_csv(save_name_csv, index=False)

    #make and save timeseries plot
    plt.plot(df['datetime'], df['distances'], '-o')
    plt.xlabel('Time (UTC)')
    plt.ylabel('Cross-Shore Position (m)')
    plt.xticks(rotation=90)
    plt.xlim(min(df['datetime']), max(df['datetime']))
    plt.ylim(min(df['distances']), max(df['distances']))
    plt.minorticks_on()
    plt.tight_layout()
    plt.savefig(save_name_png, dpi=300)
    plt.close()

    nao.plot_ts_with_nao(save_name_csv)
    tsa.main(save_name_csv)
    lsp.plot_timeseries_with_fit(save_name_csv, projection=10)
    rm.plot_timeseries_with_rolling_means_and_linear_fit(save_name_csv, projection=10)
def batch_transect_timeseries(shorelines,
                              transects,
                              sitename,
                              output_folder,
                              switch_dir=False):
    """
    Generates timeseries of shoreline cross-shore position
    given a shapefile containing shorelines and a shapefile containing
    cross-shore transects. Computes interesection points between shorelines
    and transects. Uses earliest shoreline intersection point as origin.
    
    inputs:
    shoreline_shapefile (str): path to shapefile containing shorelines
                               (needs a field 'datetime' YYYY-MM-DD-HH-MM)
    transect_shapefile (str): path to shapefile containing cross-shore transects
    sitename (str): name of site
    output_folder (str): path to save csvs and png figures to
    switch_dir (optional): default is False, set to True if transects are in the opposite direction
    """
    transects = gpd.read_file(transects)
    transects = transects.reset_index()
    for i in range(len(transects)):
        transect = transects[transects['index']==i]
        transect = transect.reset_index()
        transect_timeseries(shorelines,
                            transect,
                            sitename,
                            i,
                            output_folder,
                            switch_dir=switch_dir,
                            batch=True)


