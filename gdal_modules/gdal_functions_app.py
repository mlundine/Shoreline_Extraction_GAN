# -*- coding: utf-8 -*-
"""
Created on Wed Apr  8 10:37:12 2020

@author: Mark Lundine
"""
import os
from osgeo import gdal, ogr, osr
import osgeo.gdalnumeric as gdn
import numpy as np
import glob
import pandas as pd
import cv2

def gdal_open(image_path):
    ### read in image to classify with gdal
    driverTiff = gdal.GetDriverByName('GTiff')
    input_raster = gdal.Open(image_path)
    nbands = input_raster.RasterCount
    prj = input_raster.GetProjection()
    gt = input_raster.GetGeoTransform()
    ### create an empty array, each column of the empty array will hold one band of data from the image
    ### loop through each band in the image nad add to the data array
    data = np.empty((input_raster.RasterYSize, input_raster.RasterXSize, nbands))
    for i in range(1, nbands+1):
        band = input_raster.GetRasterBand(i).ReadAsArray()
        data[:, :, i-1] = band
    input_raster = None
    return data, prj, gt

def gdal_get_coords_and_res_path(folder, saveFile):
    """
    Takes a folder of geotiffs and outputs a csv with bounding box coordinates and x and y resolution
    inputs:
    folder (string): filepath to folder of geotiffs
    saveFile (string): filepath to csv to save to
    """
    dems = glob.glob(folder + '/*.tif')
    myList = [None]*(len(dems)+1)
    myList[0] = ['file', 'xmin', 'ymin', 'xmax', 'ymax', 'xres', 'yres', 'epsg', 'cols', 'rows']
    i=1
    for dem in dems:
        src = gdal.Open(dem)
        proj = osr.SpatialReference(wkt=src.GetProjection())
        epsg = int(proj.GetAttrValue('AUTHORITY',1))
        cols, rows = src.RasterXSize, src.RasterYSize
        xmin, xres, xskew, ymax, yskew, yres  = src.GetGeoTransform()
        xmax = xmin + (src.RasterXSize * xres)
        ymin = ymax + (src.RasterYSize * yres)
        myList[i]=[dem, xmin, ymin, xmax, ymax, xres, -yres, epsg, cols, rows]
        src = None
        i=i+1
    np.savetxt(saveFile, myList, delimiter=",", fmt='%s')
    df = pd.read_csv(saveFile)
    num_images = len(df)
    return num_images, saveFile

def gdal_get_coords_and_res_list(dems, saveFile):
    """
    Takes a folder of geotiffs and outputs a csv with bounding box coordinates and x and y resolution
    inputs:
    folder (string): list of geotiff filepaths
    saveFile (string): filepath to csv to save to
    """
    myList = [None]*(len(dems)+1)
    myList[0] = ['file', 'xmin', 'ymin', 'xmax', 'ymax', 'xres', 'yres', 'epsg', 'cols', 'rows']
    i=1
    for dem in dems:
        src = gdal.Open(dem)
        proj = osr.SpatialReference(wkt=src.GetProjection())
        epsg = int(proj.GetAttrValue('AUTHORITY',1))
        cols, rows = src.RasterXSize, src.RasterYSize
        xmin, xres, xskew, ymax, yskew, yres  = src.GetGeoTransform()
        xmax = xmin + (src.RasterXSize * xres)
        ymin = ymax + (src.RasterYSize * yres)
        myList[i]=[dem, xmin, ymin, xmax, ymax, xres, -yres, epsg, cols, rows]
        src = None
        i=i+1
    np.savetxt(saveFile, myList, delimiter=",", fmt='%s')
    df = pd.read_csv(saveFile)
    num_images = len(df)
    return num_images, saveFile

def gdal_convert(inFolder, outFolder, inType, outType, size=256):
    """
    Converts geotiffs and erdas imagine images to .tif,.jpg, .png, or .img
    inputs:
    inFolder (string): folder of .tif or .img images
    outFolder (string): folder to save result to
    inType (string): extension of input images ('.tif' or '.img')
    outType (string): extension of output images ('.tif', '.img', '.jpg', '.png')
    """
    
    for im in glob.glob(inFolder + '/*'+inType):
        imName = os.path.splitext(os.path.basename(im))[0]
        outIm = os.path.join(outFolder, imName+outType)
        if outType == '.npy':
            raster = gdal.Open(im)
            bands = [raster.GetRasterBand(i) for i in range(1, raster.RasterCount+1)]
            arr = np.array([gdn.BandReadAsArray(band) for band in bands]).astype('float32')
            arr = np.transpose(arr, [1,2,0])
            np.save(outIm, arr)
            raster = None
        if outType == '.jpeg':
            raster = gdal.Open(im)
            bands = [raster.GetRasterBand(i) for i in range(1, raster.RasterCount+1)]
            arr = np.array([gdn.BandReadAsArray(band) for band in bands]).astype('float32')
            arr = np.transpose(arr, [1,2,0])
            if np.shape(arr)[0]!=size or np.shape(arr)[1]!=size:
                raster = None
                arr = None
                continue
            stats = [raster.GetRasterBand(i).GetStatistics(True, True) for i in range(1,raster.RasterCount+1)]
            vmin, vmax, vmean, vstd = zip(*stats)
            if raster.RasterCount > 1:
                bandList = [1,2,3]
            else:
                bandList = [1]
            gdal.Translate(outIm, raster, scaleParams = list(zip(*[vmin, vmax])), bandList = bandList)
            raster = None
            arr = None
        else:
            raster = gdal.Open(im)
            gdal.Translate(outIm, raster)
            raster = None
            
def raster_to_polygon(raster_path):
    """
    Converts raster with discrete pixel values to polygons
    inputs:
    raster_path: the input raster filepath
    """
    module = os.path.join(os.getcwd(), 'gdal_modules', 'gdal_polygonize.py')
    shape_path = os.path.splitext(raster_path)[0]+'poly.shp'
    os.system('python ' + module + ' ' + raster_path + ' ' + shape_path)

def raster_to_polygon_batch(folder):
    """
    Converts a folder of rasters to shapefiles
    inputs:
    folder: filepath to folder of geotiffs
    """
    for raster in glob.glob(folder + '/*.tif'):
        raster_to_polygon(raster)

def mergeShapes(folder, outShape):
    """
    Merges a bunch of shapefiles. Sshapefiles have to have same fields
    in attribute table.
    inputs:
    folder: filepath to folder with all of the shapefiles
    outShape: filepath to file to save to, has to have .shp extension.
    """
    module = os.path.join(os.getcwd(), 'gdal_modules', 'MergeSHPfiles_cmd.py')
    os.system('python '+ module + ' ' + folder + ' ' + outShape)


def delete_empty_images(path_to_folder):
    """
    deletes geotiffs that are all zeros, good for cleaning up tiling results
    inputs:
    path_to_folder (str): filepath to the folder with images that you want to delete
    """
    for image in glob.glob(path_to_folder + '/*.tif'):
        array = gdal_open(image)[0]
        no_data = len(np.unique(array))
        if no_data < 5:
            cmd = 'gdalmanage delete ' + image
            os.system(cmd)
        array = None
          
    
