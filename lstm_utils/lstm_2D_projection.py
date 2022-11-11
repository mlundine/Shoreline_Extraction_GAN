import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
import shapely
import geopandas as gpd
import glob
from math import degrees, atan2, radians

def gb(x1, y1, x2, y2):
    angle = degrees(atan2(y2 - y1, x2 - x1))
    bearing = angle
    #bearing = (90 - angle) % 360
    return bearing

def single_transect(projection_df_path, transect_id, transect_shp_path, firstX, firstY, switch_dir=False):
    ##Open file, assign variables
    projection_df = pd.read_csv(projection_df_path)
    means = projection_df['forecast_mean_position']
    uppers = projection_df['forecast_upper_conf']
    lowers = projection_df['forecast_lower_conf']

    ##Get the angle of the transect
    transect_df = gpd.read_file(transect_shp_path)
    transect_df = transect_df.reset_index()
    transect = transect_df.iloc[transect_id]
    first, last = transect['geometry'].boundary
    if switch_dir == False:
        angle = radians(gb(first.x, first.y, last.x, last.y))
    else:
        angle = radians(gb(last.x, last.y, first.x, first.y))
    ##Compute Projected X,Y positions
    northings_mean = firstY + means*np.sin(angle)
    eastings_mean = firstX + means*np.cos(angle)
    northings_upper = firstY + uppers*np.sin(angle)
    eastings_upper = firstX + uppers*np.cos(angle)
    northings_lower = firstY + lowers*np.sin(angle)
    eastings_lower = firstX + lowers*np.cos(angle)

    ##Put positions into dataframe
    projection_df['northings_mean'] = northings_mean
    projection_df['eastings_mean'] = eastings_mean
    projection_df['northings_upper'] = northings_upper
    projection_df['eastings_upper'] = eastings_upper
    projection_df['northings_lower'] = northings_lower
    projection_df['eastings_lower'] = eastings_lower

    ##Save new dataframe
    projection_df.to_csv(projection_df_path, index=False)

    ##Clean Up
    projection_df = None
    means = None
    uppers = None
    lowers = None
    northings_mean = None
    eastings_mean = None
    northings_upper = None
    eastings_upper = None
    northings_lower = None
    eastings_lower = None
    
    return projection_df_path

def lists_to_Polygon(list1, list2):
    new_list = [None]*len(list1)
    for i in range(len(list1)):
        new_list[i] = (list1[i], list2[i])
    points = [None]*len(new_list)
    for i in range(len(new_list)):
        x,y = new_list[i]
        point = shapely.geometry.Point(x,y)
        points[i] = point
    polygon = shapely.geometry.Polygon(points)
    return polygon

def lists_to_LineString(list1, list2):
    new_list = [None]*len(list1)
    for i in range(len(list1)):
        new_list[i] = (list1[i], list2[i])
    points = [None]*len(new_list)
    for i in range(len(new_list)):
        x,y = new_list[i]
        point = shapely.geometry.Point(x,y)
        points[i] = point
    line = shapely.geometry.LineString(points)
    return line

def multiple_transects(projection_df_path_list,
                       extracted_df_path_list,
                       transect_ids,
                       transect_shp_path,
                       projection_times,
                       savefolder,
                       sitename,
                       epsg,
                       switch_dir=False):
    """
    Build two shapefiies
    One that contains mean LSTM shoreline projections (so lines with a timestamp)
    ONe that contains confidence interval polygons (polygons with a timestamp)
    """
    mean_savepath = os.path.join(savefolder, sitename+'_mean_shorelines.shp')
    conf_savepath = os.path.join(savefolder, sitename+'_confidence_intervals.shp')
    time = projection_times
    
    new_proj_path_list = [None]*len(projection_df_path_list)
    for i in range(len(projection_df_path_list)):
        projection_df_path = projection_df_path_list[i]
        print(os.path.basename(projection_df_path))
        transect_id = transect_ids[i]
        extracted_df = pd.read_csv(extracted_df_path_list[i])
        firstX = extracted_df['eastings'][0]
        firstY = extracted_df['northings'][0]
        new_projection_df_path = single_transect(projection_df_path, transect_id, transect_shp_path, firstX, firstY, switch_dir=switch_dir)
        new_proj_path_list[i] = new_projection_df_path

    ###Should have length of projected time
    shapefile_mean_dict = {'Timestamp':time}
    shapefile_mean_df = pd.DataFrame(shapefile_mean_dict)
    shapefile_mean_geoms = [None]*len(shapefile_mean_df)

    shapefile_confidence_intervals_dict = {'Timestamp':time}
    shapefile_confidence_intervals_df = pd.DataFrame(shapefile_confidence_intervals_dict)
    shapefile_confidence_intervals_geoms = [None]*len(shapefile_confidence_intervals_df)
    ###Loop over projected time
    for i in range(len(time)):
        ###Make empty lists to hold mean coordinates, upper and lower conf coordinates
        ###These are for one time
        shoreline_eastings = [None]*len(transect_ids)
        shoreline_northings = [None]*len(transect_ids)
        shoreline_eastings_upper = [None]*len(transect_ids)
        shoreline_northings_upper = [None]*len(transect_ids)
        shoreline_eastings_lower = [None]*len(transect_ids)
        shoreline_northings_lower = [None]*len(transect_ids)
        timestamp = [time[i]]*len(transect_ids)
        for j in range(len(new_proj_path_list)):
            transect_id = transect_ids[j]
            proj_path = new_proj_path_list[j]
            proj_df = pd.read_csv(proj_path)
            shoreline_eastings[j] = proj_df['eastings_mean'][i]
            shoreline_northings[j] = proj_df['northings_mean'][i]
            shoreline_eastings_upper[j] = proj_df['eastings_upper'][i]
            shoreline_northings_upper[j] = proj_df['northings_upper'][i]
            shoreline_eastings_lower[j] = proj_df['eastings_lower'][i]
            shoreline_northings_lower[j] = proj_df['northings_lower'][i]
        confidence_interval_x = np.concatenate((shoreline_eastings_upper, list(reversed(shoreline_eastings_lower))))
        confidence_interval_y = np.concatenate((shoreline_northings_upper, list(reversed(shoreline_northings_lower))))
        
        confidence_interval_polygon = lists_to_Polygon(confidence_interval_x, confidence_interval_y)
        shapefile_confidence_intervals_geoms[i] = confidence_interval_polygon
        
        mean_shoreline_line = lists_to_LineString(shoreline_eastings, shoreline_northings)
        shapefile_mean_geoms[i] = mean_shoreline_line
        
    shapefile_mean_geodf = gpd.GeoDataFrame(shapefile_mean_df, geometry = shapefile_mean_geoms)
    shapefile_mean_geodf = shapefile_mean_geodf.set_crs('epsg:'+str(epsg))
    shapefile_confidence_intervals_geodf = gpd.GeoDataFrame(shapefile_confidence_intervals_df, geometry = shapefile_confidence_intervals_geoms)
    shapefile_confidence_intervals_geodf = shapefile_confidence_intervals_geodf.set_crs('epsg:'+str(epsg))
    
    shapefile_mean_geodf.to_file(mean_savepath)
    shapefile_confidence_intervals_geodf.to_file(conf_savepath)

def main(sitename,
         transect_id_range,
         projected_folder,
         extracted_folder,
         save_folder,
         transect_shp_path,
         epsg,
         switch_dir=False):
    """
    Takes projected cross-shore positions and uncertainties and constructs 2D projected shorelines/uncertainties
    Saves these to two shapefiles (mean shorelines and confidence intervals)
    inputs:
    sitename: Name of site (str)
    transect_id_range: number of transects (int)
    projected_folder: path to folder containing projected data (str)
    extracted_folder: path to folder containing extracted shoreline data (str)
    save_folder: folder to save projected shoreline shapefiles to (str)
    transect_shp_path: path to shapefile containing transects (str)
    epsg: epsg code (int)
    switch_dir: Optional, if True, then transect direction is reversed
    """
    transect_ids = range(transect_id_range)
    projection_df_path_list = [None]*len(transect_ids)
    extracted_df_path_list = [None]*len(projection_df_path_list)
    for j in range(len(transect_ids)):
        i = transect_ids[j]
        projected = os.path.join(projected_folder, sitename+'_'+str(i)+'project.csv')
        extracted = os.path.join(extracted_folder, sitename+'_'+str(i)+'.csv')
        projection_df_path_list[j] = projected
        extracted_df_path_list[j] = extracted

    projection_times = pd.read_csv(projection_df_path_list[0])['time']

    multiple_transects(projection_df_path_list,
                       extracted_df_path_list,
                       transect_ids,
                       transect_shp_path,
                       projection_times,
                       save_folder,
                       sitename,
                       epsg,
                       switch_dir=False)
    
##main('CapeHenlopen',
##     235,
##     r'D:\Shoreline_Extraction_GAN\model_outputs\processed\CapeHenlopenNew\projected',
##     r'D:\Shoreline_Extraction_GAN\model_outputs\processed\CapeHenlopenNew\transects',
##     r'D:\Shoreline_Extraction_GAN\model_outputs\processed\CapeHenlopenNew\projected',
##     r'D:\Shoreline_Extraction_GAN\model_outputs\processed\CapeHenlopenNew\CapeHenlopen_reference_shoreline_transects_15m.shp',
##     32618,
##     )
