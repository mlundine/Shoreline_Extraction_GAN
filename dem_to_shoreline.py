# Mark Lundine

import numpy as np
from osgeo import osr
from osgeo import ogr
from osgeo import gdal
import geopandas as gpd


def lidar_dem_to_shoreline(dem_path,
                           contour_path,
                           shoreline_path,
                           no_data_value=-9999,
                           filter_extra=False):
    """
    Uses gdal to take a raster DEM (.tif, .img, Esri grids, etc.)
    and generate contours. Then extracts the 0 contour as the shoreline.
    If data includes areas that have low spots other than the shore
    (ex: ponds inland from dune), then can filter out the longest 0 contour as shoreline.

    inputs:
    dem_path: path to the dem (str)
    contour_path: path to save all of the contours to, end this with .shp (str)
    shoreline_path: path to save the 0 contour to, end with .shp (str)
    no_data_value (optional, default=-9999): the no data value for the raster 
    filter_extra (optional, default=False): If set to True, will only save the longest 0 contour to the shoreline file

    #### example    
    lidar_dem_to_shoreline(r'cape_henlopen_lidar_1m.tif',
                           r'cape_contours.shp',
                           r'cape_shoreline.shp',
                           no_data_value=-9999,
                           filter_extra=True)
    """
    # Open tif file as select band
    raster = gdal.Open(dem_path)

    # loop through the image bands
    for i in range(1, raster.RasterCount + 1):
        # set the nodata value of the band
        raster.GetRasterBand(i).SetNoDataValue(no_data_value)

    # read first band and projection
    first_band = raster.GetRasterBand(1)
    proj = osr.SpatialReference(wkt=raster.GetProjection())        

    # set up contour shapefile path
    contour_ds = ogr.GetDriverByName("ESRI Shapefile").CreateDataSource(contour_path)

    # define layer name and spatial 
    contour_shp = contour_ds.CreateLayer('contour', proj)

    # define fields of id and elev
    fieldDef = ogr.FieldDefn("ID", ogr.OFTInteger)
    contour_shp.CreateField(fieldDef)
    fieldDef = ogr.FieldDefn("elev", ogr.OFTReal)
    contour_shp.CreateField(fieldDef)

    # Write shapefile
    # ContourGenerate(Band srcBand, double contourInterval,
    #                 double contourBase, int fixedLevelCount,
    #                 int useNoData, double noDataValue, 
    #                 Layer dstLayer, int idField, int elevField)
    # Can change contourInterval to get finer/coarser contour resolution, can also change contourBase to change
    # which elevation to start at. Right now it is set at a CI of 1m, and a base of 0m.
    gdal.ContourGenerate(first_band, 1.0, 0.0, [], 1, no_data_value, 
                         contour_shp, 0, 1)

    contour_ds.Destroy()

    # filter out 0m contour (shoreline) with geopandas
    contour_shapefile = gpd.read_file(contour_path)
    shoreline = contour_shapefile[contour_shapefile['elev']==0]
    shoreline['shore_len'] = shoreline['geometry'].length

    # filter out longest 0m contour
    if filter_extra == True:
        lengths = shoreline['shore_len']
        max_length = max(lengths)
        filtered_shoreline = shoreline[shoreline['shore_len']==max_length]
        filtered_shoreline.to_file(shoreline_path)
    else:
        shoreline.to_file(shoreline_path)


