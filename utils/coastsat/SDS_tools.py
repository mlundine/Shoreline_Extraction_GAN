"""
This module contains utilities to work with satellite images
    
Author: Kilian Vos, Water Research Laboratory, University of New South Wales
"""

# load modules
import os
import numpy as np
import matplotlib.pyplot as plt
import pdb

# other modules
from osgeo import gdal, osr
import geopandas as gpd
from shapely import geometry
import skimage.transform as transform
from astropy.convolution import convolve

###################################################################################################
# COORDINATES CONVERSION FUNCTIONS
###################################################################################################

def convert_pix2world(points, georef):
    """
    Converts pixel coordinates (pixel row and column) to world projected 
    coordinates performing an affine transformation.
    
    KV WRL 2018

    Arguments:
    -----------
    points: np.array or list of np.array
        array with 2 columns (row first and column second)
    georef: np.array
        vector of 6 elements [Xtr, Xscale, Xshear, Ytr, Yshear, Yscale]
                
    Returns:    
    -----------
    points_converted: np.array or list of np.array 
        converted coordinates, first columns with X and second column with Y
        
    """
    
    # make affine transformation matrix
    aff_mat = np.array([[georef[1], georef[2], georef[0]],
                       [georef[4], georef[5], georef[3]],
                       [0, 0, 1]])
    # create affine transformation
    tform = transform.AffineTransform(aff_mat)

    # if list of arrays
    if type(points) is list:
        points_converted = []
        # iterate over the list
        for i, arr in enumerate(points): 
            tmp = arr[:,[1,0]]
            points_converted.append(tform(tmp))
          
    # if single array
    elif type(points) is np.ndarray:
        tmp = points[:,[1,0]]
        points_converted = tform(tmp)
        
    else:
        raise Exception('invalid input type')
        
    return points_converted

def convert_world2pix(points, georef):
    """
    Converts world projected coordinates (X,Y) to image coordinates 
    (pixel row and column) performing an affine transformation.
    
    KV WRL 2018

    Arguments:
    -----------
    points: np.array or list of np.array
        array with 2 columns (X,Y)
    georef: np.array
        vector of 6 elements [Xtr, Xscale, Xshear, Ytr, Yshear, Yscale]
                
    Returns:    
    -----------
    points_converted: np.array or list of np.array 
        converted coordinates (pixel row and column)
    
    """
    
    # make affine transformation matrix
    aff_mat = np.array([[georef[1], georef[2], georef[0]],
                       [georef[4], georef[5], georef[3]],
                       [0, 0, 1]])
    # create affine transformation
    tform = transform.AffineTransform(aff_mat)
    
    # if list of arrays
    if type(points) is list:
        points_converted = []
        # iterate over the list
        for i, arr in enumerate(points): 
            points_converted.append(tform.inverse(points))
            
    # if single array    
    elif type(points) is np.ndarray:
        points_converted = tform.inverse(points)
        
    else:
        print('invalid input type')
        raise
        
    return points_converted


def convert_epsg(points, epsg_in, epsg_out):
    """
    Converts from one spatial reference to another using the epsg codes
    
    KV WRL 2018

    Arguments:
    -----------
    points: np.array or list of np.ndarray
        array with 2 columns (rows first and columns second)
    epsg_in: int
        epsg code of the spatial reference in which the input is
    epsg_out: int
        epsg code of the spatial reference in which the output will be            
                
    Returns:    
    -----------
    points_converted: np.array or list of np.array 
        converted coordinates from epsg_in to epsg_out
        
    """
    
    # define input and output spatial references
    inSpatialRef = osr.SpatialReference()
    inSpatialRef.ImportFromEPSG(epsg_in)
    outSpatialRef = osr.SpatialReference()
    outSpatialRef.ImportFromEPSG(epsg_out)
    # create a coordinates transform
    coordTransform = osr.CoordinateTransformation(inSpatialRef, outSpatialRef)
    # if list of arrays
    if type(points) is list:
        points_converted = []
        # iterate over the list
        for i, arr in enumerate(points): 
            points_converted.append(np.array(coordTransform.TransformPoints(arr)))
    # if single array
    elif type(points) is np.ndarray:
        points_converted = np.array(coordTransform.TransformPoints(points))  
    else:
        raise Exception('invalid input type')

    return points_converted

###################################################################################################
# IMAGE ANALYSIS FUNCTIONS
###################################################################################################
    
def nd_index(im1, im2, cloud_mask):
    """
    Computes normalised difference index on 2 images (2D), given a cloud mask (2D).

    KV WRL 2018

    Arguments:
    -----------
    im1: np.array
        first image (2D) with which to calculate the ND index
    im2: np.array
        second image (2D) with which to calculate the ND index
    cloud_mask: np.array
        2D cloud mask with True where cloud pixels are

    Returns:    
    -----------
    im_nd: np.array
        Image (2D) containing the ND index
        
    """

    # reshape the cloud mask
    vec_mask = cloud_mask.reshape(im1.shape[0] * im1.shape[1])
    # initialise with NaNs
    vec_nd = np.ones(len(vec_mask)) * np.nan
    # reshape the two images
    vec1 = im1.reshape(im1.shape[0] * im1.shape[1])
    vec2 = im2.reshape(im2.shape[0] * im2.shape[1])
    # compute the normalised difference index
    temp = np.divide(vec1[~vec_mask] - vec2[~vec_mask],
                     vec1[~vec_mask] + vec2[~vec_mask])
    vec_nd[~vec_mask] = temp
    # reshape into image
    im_nd = vec_nd.reshape(im1.shape[0], im1.shape[1])

    return im_nd
    
def image_std(image, radius):
    """
    Calculates the standard deviation of an image, using a moving window of 
    specified radius. Uses astropy's convolution library'
    
    Arguments:
    -----------
    image: np.array
        2D array containing the pixel intensities of a single-band image
    radius: int
        radius defining the moving window used to calculate the standard deviation. 
        For example, radius = 1 will produce a 3x3 moving window.
        
    Returns:    
    -----------
    win_std: np.array
        2D array containing the standard deviation of the image
        
    """  
    
    # convert to float
    image = image.astype(float)
    # first pad the image
    image_padded = np.pad(image, radius, 'reflect')
    # window size
    win_rows, win_cols = radius*2 + 1, radius*2 + 1
    # calculate std with uniform filters
    win_mean = convolve(image_padded, np.ones((win_rows,win_cols)), boundary='extend',
                        normalize_kernel=True, nan_treatment='interpolate', preserve_nan=True)
    win_sqr_mean = convolve(image_padded**2, np.ones((win_rows,win_cols)), boundary='extend',
                        normalize_kernel=True, nan_treatment='interpolate', preserve_nan=True)
    win_var = win_sqr_mean - win_mean**2
    win_std = np.sqrt(win_var)
    # remove padding
    win_std = win_std[radius:-radius, radius:-radius]

    return win_std

def mask_raster(fn, mask):
    """
    Masks a .tif raster using GDAL.
    
    Arguments:
    -----------
    fn: str
        filepath + filename of the .tif raster
    mask: np.array
        array of boolean where True indicates the pixels that are to be masked
        
    Returns:    
    -----------
    Overwrites the .tif file directly
        
    """ 
    
    # open raster
    raster = gdal.Open(fn, gdal.GA_Update)
    # mask raster
    for i in range(raster.RasterCount):
        out_band = raster.GetRasterBand(i+1)
        out_data = out_band.ReadAsArray()
        out_band.SetNoDataValue(0)
        no_data_value = out_band.GetNoDataValue()
        out_data[mask] = no_data_value
        out_band.WriteArray(out_data)
    # close dataset and flush cache
    raster = None


###################################################################################################
# UTILITIES
###################################################################################################
    
def get_filepath(inputs,satname):
    """
    Create filepath to the different folders containing the satellite images.
    
    KV WRL 2018

    Arguments:
    -----------
    inputs: dict with the following keys
        'sitename': str
            name of the site
        'polygon': list
            polygon containing the lon/lat coordinates to be extracted,
            longitudes in the first column and latitudes in the second column,
            there are 5 pairs of lat/lon with the fifth point equal to the first point:
            ```
            polygon = [[[151.3, -33.7],[151.4, -33.7],[151.4, -33.8],[151.3, -33.8],
            [151.3, -33.7]]]
            ```
        'dates': list of str
            list that contains 2 strings with the initial and final dates in 
            format 'yyyy-mm-dd':
            ```
            dates = ['1987-01-01', '2018-01-01']
            ```
        'sat_list': list of str
            list that contains the names of the satellite missions to include: 
            ```
            sat_list = ['L5', 'L7', 'L8', 'S2']
            ```
        'filepath_data': str
            filepath to the directory where the images are downloaded
    satname: str
        short name of the satellite mission ('L5','L7','L8','S2')
                
    Returns:    
    -----------
    filepath: str or list of str
        contains the filepath(s) to the folder(s) containing the satellite images
    
    """     
    
    sitename = inputs['sitename']
    filepath_data = inputs['filepath']
    # access the images
    if satname == 'L5':
        # access downloaded Landsat 5 images
        filepath = os.path.join(filepath_data, sitename, satname, '30m')
    elif satname == 'L7':
        # access downloaded Landsat 7 images
        filepath_pan = os.path.join(filepath_data, sitename, 'L7', 'pan')
        filepath_ms = os.path.join(filepath_data, sitename, 'L7', 'ms')
        filepath = [filepath_pan, filepath_ms]
    elif satname == 'L8':
        # access downloaded Landsat 8 images
        filepath_pan = os.path.join(filepath_data, sitename, 'L8', 'pan')
        filepath_ms = os.path.join(filepath_data, sitename, 'L8', 'ms')
        filepath = [filepath_pan, filepath_ms]
    elif satname == 'S2':
        # access downloaded Sentinel 2 images
        filepath10 = os.path.join(filepath_data, sitename, satname, '10m')
        filepath20 = os.path.join(filepath_data, sitename, satname, '20m')
        filepath60 = os.path.join(filepath_data, sitename, satname, '60m')
        filepath = [filepath10, filepath20, filepath60]
            
    return filepath
    
def get_filenames(filename, filepath, satname):
    """
    Creates filepath + filename for all the bands belonging to the same image.
    
    KV WRL 2018

    Arguments:
    -----------
    filename: str
        name of the downloaded satellite image as found in the metadata
    filepath: str or list of str
        contains the filepath(s) to the folder(s) containing the satellite images
    satname: str
        short name of the satellite mission       
        
    Returns:    
    -----------
    fn: str or list of str
        contains the filepath + filenames to access the satellite image
        
    """     
    
    if satname == 'L5':
        fn = os.path.join(filepath, filename)
    if satname == 'L7' or satname == 'L8':
        filename_ms = filename.replace('pan','ms')
        fn = [os.path.join(filepath[0], filename),
              os.path.join(filepath[1], filename_ms)]
    if satname == 'S2':
        filename20 = filename.replace('10m','20m')
        filename60 = filename.replace('10m','60m')
        fn = [os.path.join(filepath[0], filename),
              os.path.join(filepath[1], filename20),
              os.path.join(filepath[2], filename60)]
        
    return fn

def merge_output(output):
    """
    Function to merge the output dictionnary, which has one key per satellite mission
    into a dictionnary containing all the shorelines and dates ordered chronologically.
    
    Arguments:
    -----------
    output: dict
        contains the extracted shorelines and corresponding dates, organised by 
        satellite mission
    
    Returns:    
    -----------
    output_all: dict
        contains the extracted shorelines in a single list sorted by date
    
    """     
    
    # initialize output dict
    output_all = dict([])
    satnames = list(output.keys())
    for key in output[satnames[0]].keys():
        output_all[key] = []
    # create extra key for the satellite name
    output_all['satname'] = []
    # fill the output dict
    for satname in list(output.keys()):
        for key in output[satnames[0]].keys():
            output_all[key] = output_all[key] + output[satname][key]
        output_all['satname'] = output_all['satname'] + [_ for _ in np.tile(satname,
                  len(output[satname]['dates']))]
    # sort chronologically
    idx_sorted = sorted(range(len(output_all['dates'])), key=output_all['dates'].__getitem__)
    for key in output_all.keys():
        output_all[key] = [output_all[key][i] for i in idx_sorted]

    return output_all


def remove_duplicates(output):
    """
    Function to remove from the output dictionnary entries containing shorelines for 
    the same date and satellite mission. This happens when there is an overlap between 
    adjacent satellite images.

    Arguments:
    -----------
        output: dict
            contains output dict with shoreline and metadata

    Returns:
    -----------
        output_no_duplicates: dict
            contains the updated dict where duplicates have been removed

    """

    # nested function
    def duplicates_dict(lst):
        "return duplicates and indices"
        def duplicates(lst, item):
                return [i for i, x in enumerate(lst) if x == item]
        return dict((x, duplicates(lst, x)) for x in set(lst) if lst.count(x) > 1)

    dates = output['dates']
    # make a list with year/month/day
    dates_str = [_.strftime('%Y%m%d') for _ in dates]
    # create a dictionnary with the duplicates
    dupl = duplicates_dict(dates_str)
    # if there are duplicates, only keep the first element
    if dupl:
        output_no_duplicates = dict([])
        idx_remove = []
        for k,v in dupl.items():
            idx_remove.append(v[0])
        idx_remove = sorted(idx_remove)
        idx_all = np.linspace(0, len(dates_str)-1, len(dates_str))
        idx_keep = list(np.where(~np.isin(idx_all,idx_remove))[0])
        for key in output.keys():
            output_no_duplicates[key] = [output[key][i] for i in idx_keep]
        print('%d duplicates' % len(idx_remove))
        return output_no_duplicates
    else:
        print('0 duplicates')
        return output

def remove_inaccurate_georef(output, accuracy):
    """
    Function to remove from the output dictionnary entries containing shorelines 
    that were mapped on images with inaccurate georeferencing:
        - RMSE > accuracy for Landsat images
        - failed geometric test for Sentinel images (flagged with -1)

    Arguments:
    -----------
        output: dict
            contains the extracted shorelines and corresponding metadata
        accuracy: int
            minimum horizontal georeferencing accuracy (metres) for a shoreline to be accepted

    Returns:
    -----------
        output_filtered: dict
            contains the updated dictionnary

    """

    # find indices of shorelines to be removed
    idx = np.where(~np.logical_or(np.array(output['geoaccuracy']) == -1,
                                  np.array(output['geoaccuracy']) >= accuracy))[0]
    # idx = np.where(~(np.array(output['geoaccuracy']) >= accuracy))[0]
    output_filtered = dict([])
    for key in output.keys():
        output_filtered[key] = [output[key][i] for i in idx]
    print('%d bad georef' % (len(output['geoaccuracy']) - len(idx)))
    return output_filtered

###################################################################################################
# CONVERSIONS FROM DICT TO GEODATAFRAME AND READ/WRITE GEOJSON
###################################################################################################
    
def polygon_from_kml(fn):
    """
    Extracts coordinates from a .kml file.
    
    KV WRL 2018

    Arguments:
    -----------
    fn: str
        filepath + filename of the kml file to be read          
                
    Returns:    
    -----------
    polygon: list
        coordinates extracted from the .kml file
        
    """    
    
    # read .kml file
    with open(fn) as kmlFile:
        doc = kmlFile.read() 
    # parse to find coordinates field
    str1 = '<coordinates>'
    str2 = '</coordinates>'
    subdoc = doc[doc.find(str1)+len(str1):doc.find(str2)]
    coordlist = subdoc.split('\n')
    # read coordinates
    polygon = []
    for i in range(1,len(coordlist)-1):
        polygon.append([float(coordlist[i].split(',')[0]), float(coordlist[i].split(',')[1])])
        
    return [polygon]

def transects_from_geojson(filename):
    """
    Reads transect coordinates from a .geojson file.
    
    Arguments:
    -----------
    filename: str
        contains the path and filename of the geojson file to be loaded
        
    Returns:    
    -----------
    transects: dict
        contains the X and Y coordinates of each transect
        
    """  
    
    gdf = gpd.read_file(filename)
    transects = dict([])
    for i in gdf.index:
        transects[gdf.loc[i,'name']] = np.array(gdf.loc[i,'geometry'].coords)
        
    print('%d transects have been loaded' % len(transects.keys()))

    return transects

def output_to_gdf(output):
    """
    Saves the mapped shorelines as a gpd.GeoDataFrame    
    
    KV WRL 2018

    Arguments:
    -----------
    output: dict
        contains the coordinates of the mapped shorelines + attributes          
                
    Returns:    
    -----------
    gdf_all: gpd.GeoDataFrame
        contains the shorelines + attirbutes
  
    """    
     
    # loop through the mapped shorelines
    counter = 0
    for i in range(len(output['shorelines'])):
        # skip if there shoreline is empty 
        if len(output['shorelines'][i]) == 0:
            continue
        else:
            # save the geometry + attributes
            coords = output['shorelines'][i]
            geom = geometry.MultiPoint([(coords[_,0], coords[_,1]) for _ in range(coords.shape[0])])
            gdf = gpd.GeoDataFrame(geometry=gpd.GeoSeries(geom))
            gdf.index = [i]
            gdf.loc[i,'date'] = output['dates'][i].strftime('%Y-%m-%d %H:%M:%S')
            gdf.loc[i,'satname'] = output['satname'][i]
            gdf.loc[i,'geoaccuracy'] = output['geoaccuracy'][i]
            gdf.loc[i,'cloud_cover'] = output['cloud_cover'][i]
            # store into geodataframe
            if counter == 0:
                gdf_all = gdf
            else:
                gdf_all = gdf_all.append(gdf)
            counter = counter + 1
            
    return gdf_all

def transects_to_gdf(transects):
    """
    Saves the shore-normal transects as a gpd.GeoDataFrame    
    
    KV WRL 2018

    Arguments:
    -----------
    transects: dict
        contains the coordinates of the transects          
                
    Returns:    
    -----------
    gdf_all: gpd.GeoDataFrame

        
    """  
       
    # loop through the mapped shorelines
    for i,key in enumerate(list(transects.keys())):
        # save the geometry + attributes
        geom = geometry.LineString(transects[key])
        gdf = gpd.GeoDataFrame(geometry=gpd.GeoSeries(geom))
        gdf.index = [i]
        gdf.loc[i,'name'] = key
        # store into geodataframe
        if i == 0:
            gdf_all = gdf
        else:
            gdf_all = gdf_all.append(gdf)
            
    return gdf_all