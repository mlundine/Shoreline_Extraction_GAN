import os
import sys
dir_path = os.path.dirname(os.path.realpath(__file__))
gdal_modules = os.path.join(dir_path, 'gdal_modules')
coastsat = os.path.join(dir_path, 'coastsat')
sys.path.append(gdal_modules)
sys.path.append(coastsat)
import glob
import pandas as pd
import gdal_functions_app as gda
import SDS_download, SDS_preprocess2
import numpy as np
import geopandas as gpd


def get_metadata(site_folder, sitename, save_file):
    """
    gets metadata from geotiffs downloaded thru coastsat
    inputs:
    site_folder: path to site (str)
    save_file: path to csv to save metadata to (str)
    outputs:
    num_images: number of images (int)
    metadata_csv: path to the metadata csv (str)
    """
    l5 = os.path.join(site_folder, 'L5', '30m')
    l7 = os.path.join(site_folder, 'L7', 'ms')
    l8 = os.path.join(site_folder, 'L8', 'ms')
    s2 = os.path.join(site_folder, 'S2', '10m')

    try:
        l5_ims = glob.glob(l5 + '\*.tif')
    except:
        pass
    try:
        l7_ims = glob.glob(l7 + '\*.tif')
    except:
        pass     
    try:
        l8_ims = glob.glob(l8 + '\*.tif')
    except:
        pass
    try:
        s2_ims = glob.glob(s2 + '\*.tif')
    except:
        pass

    dems = list(np.concatenate((l5_ims, l7_ims, l8_ims, s2_ims)))
    num_images, metadata_csv = gda.gdal_get_coords_and_res_list(dems, save_file)

    df = pd.read_csv(metadata_csv)

    for i in range(len(df)):
        file = os.path.basename(df['file'][i])
        if file.find('_ms')>0:
            idx = file.find('_ms')
            newfile = file[0:idx]+'.tif'
            df['file'][i] = os.path.splitext(newfile)[0]
        elif file.find('_dup')>0:
            file = None
        elif file.find('_10m')>0:
            idx = file.find('_10m')
            newfile = file[0:idx]+'.tif'
            df['file'][i] = os.path.splitext(newfile)[0]
        else:
            df['file'][i] = os.path.splitext(file)[0]
    filter_df = df[~df['file'].str.contains('_dup')]

    filter_df.reset_index()
    for i in range(len(filter_df)):
        name = filter_df['file'].iloc[i]
        idx = name.find(sitename)
        if idx>0:
            new_name = name[0:idx-1]
            rest_of_name = name[idx:]
            new_name = new_name+rest_of_name
            filter_df['file'].iloc[i] = new_name
    filter_df.to_csv(metadata_csv, index=False)
    
    return num_images, metadata_csv


def download_imagery(polygon, dates, sat_list, sitename):
    """
    Downloads available satellite imagery using CoastSat download and preprocessing tools
    See https://github.com/kvos/CoastSat for original code and links to associated papers
    
    inputs:
    
    lat_long_box (list of tuples):
    latitude and longitude box [(ul_long, ul_lat),
                                (ur_long, ur_lat),
                                (ll_long, ll_lat),
                                [lr_long, lr_lat)]
                                
    sat_list: ['L5', 'L7', 'L8', 'S2'] specify L5, L7, L8, and/or S2
    
    dates: time range ['YYYY-MM-DD', 'YYYY-MM-DD']
    
    outputs:

    """
    
    # filepath where data will be stored
    filepath_data = os.path.join(os.getcwd(), 'data')

    # put all the inputs into a dictionnary
    inputs = {
        'polygon': polygon,
        'dates': dates,
        'sat_list': sat_list,
        'sitename': sitename,
        'filepath': filepath_data,
            }
        
    ### retrieve satellite images from GEE
    metadata = SDS_download.retrieve_images(inputs)

    ### if you have already downloaded the images, just load the metadata file
    metadata = SDS_download.get_metadata(inputs)   

    ### settings for cloud filtering
    settings = { 
                # general parameters:
                'cloud_thresh': 0.20,        # threshold on maximum cloud cover
                'inputs':inputs
                }
    
    ### preprocess images (cloud masking, geotiff to jpeg conversion, write geo-metadata to jpeg)
    SDS_preprocess2.save_jpg(metadata, settings)

    ##saving metadata from geotiffs to csv
    site_folder = os.path.join(filepath_data, sitename)
    metadata_csv = get_metadata(site_folder,
                                sitename,
                                os.path.join(site_folder, sitename+'.csv'))[1]
    
    return metadata_csv, site_folder


def download_from_shapefile(shapefile, basename):
    def get_points(file):
        df = gpd.read_file(file)
    
        def coord_lister(geom):
            coords = list(geom.exterior.coords)
            return (coords)
    
        coordinates = df.geometry.apply(coord_lister)
        return coordinates
    
    polygons = get_points(shapefile)
    print(polygons)
    j=1
    for poly in polygons:
        box = [[poly[0][0],poly[0][1]],
               [poly[1][0],poly[1][1]],
               [poly[2][0],poly[2][1]],
               [poly[3][0],poly[3][1]]]
        download_imagery(box,
                         ['1980-01-01', '2022-07-14'],
                         ['L7', 'L8', 'L5', 'S2'],
                         basename+str(j))
        j=j+1


