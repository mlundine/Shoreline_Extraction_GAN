"""
This module contains all the functions needed to preprocess the satellite images
 before the shorelines can be extracted. This includes creating a cloud mask and
pansharpening/downsampling the multispectral bands.

Author: Kilian Vos, Water Research Laboratory, University of New South Wales
"""

# load modules
import os
import numpy as np

# image processing modules
import skimage.transform as transform
import skimage.morphology as morphology
import sklearn.decomposition as decomposition
import skimage.exposure as exposure
import cv2

# other modules
from osgeo import gdal

# CoastSat modules
import SDS_tools

np.seterr(all='ignore') # raise/ignore divisions by 0 and nans

def create_cloud_mask(im_QA, satname, cloud_mask_issue):
    """
    Creates a cloud mask using the information contained in the QA band.

    KV WRL 2018

    Arguments:
    -----------
    im_QA: np.array
        Image containing the QA band
    satname: string
        short name for the satellite: ```'L5', 'L7', 'L8' or 'S2'```
    cloud_mask_issue: boolean
        True if there is an issue with the cloud mask and sand pixels are being
        erroneously masked on the images

    Returns:
    -----------
    cloud_mask : np.array
        boolean array with True if a pixel is cloudy and False otherwise
        
    """

    # convert QA bits (the bits allocated to cloud cover vary depending on the satellite mission)
    if satname == 'L8':
        cloud_values = [2800, 2804, 2808, 2812, 6896, 6900, 6904, 6908]
    elif satname == 'L7' or satname == 'L5' or satname == 'L4':
        cloud_values = [752, 756, 760, 764]
    elif satname == 'S2':
        cloud_values = [1024, 2048] # 1024 = dense cloud, 2048 = cirrus clouds

    # find which pixels have bits corresponding to cloud values
    cloud_mask = np.isin(im_QA, cloud_values)

    # remove cloud pixels that form very thin features. These are beach or swash pixels that are
    # erroneously identified as clouds by the CFMASK algorithm applied to the images by the USGS.
    if sum(sum(cloud_mask)) > 0 and sum(sum(~cloud_mask)) > 0:
        morphology.remove_small_objects(cloud_mask, min_size=10, connectivity=1, in_place=True)

        if cloud_mask_issue:
            elem = morphology.square(3) # use a square of width 3 pixels
            cloud_mask = morphology.binary_opening(cloud_mask,elem) # perform image opening
            # remove objects with less than 25 connected pixels
            morphology.remove_small_objects(cloud_mask, min_size=25, connectivity=1, in_place=True)

    return cloud_mask

def hist_match(source, template):
    """
    Adjust the pixel values of a grayscale image such that its histogram matches
    that of a target image.

    Arguments:
    -----------
    source: np.array
        Image to transform; the histogram is computed over the flattened
        array
    template: np.array
        Template image; can have different dimensions to source
        
    Returns:
    -----------
    matched: np.array
        The transformed output image
        
    """

    oldshape = source.shape
    source = source.ravel()
    template = template.ravel()

    # get the set of unique pixel values and their corresponding indices and
    # counts
    s_values, bin_idx, s_counts = np.unique(source, return_inverse=True,
                                            return_counts=True)
    t_values, t_counts = np.unique(template, return_counts=True)

    # take the cumsum of the counts and normalize by the number of pixels to
    # get the empirical cumulative distribution functions for the source and
    # template images (maps pixel value --> quantile)
    s_quantiles = np.cumsum(s_counts).astype(np.float64)
    s_quantiles /= s_quantiles[-1]
    t_quantiles = np.cumsum(t_counts).astype(np.float64)
    t_quantiles /= t_quantiles[-1]

    # interpolate linearly to find the pixel values in the template image
    # that correspond most closely to the quantiles in the source image
    interp_t_values = np.interp(s_quantiles, t_quantiles, t_values)

    return interp_t_values[bin_idx].reshape(oldshape)


def rescale_image_intensity(im, cloud_mask, prob_high):
    """
    Rescales the intensity of an image (multispectral or single band) by applying
    a cloud mask and clipping the prob_high upper percentile. This functions allows
    to stretch the contrast of an image, only for visualisation purposes.

    KV WRL 2018

    Arguments:
    -----------
    im: np.array
        Image to rescale, can be 3D (multispectral) or 2D (single band)
    cloud_mask: np.array
        2D cloud mask with True where cloud pixels are
    prob_high: float
        probability of exceedence used to calculate the upper percentile

    Returns:
    -----------
    im_adj: np.array
        rescaled image
    """

    # lower percentile is set to 0
    prc_low = 0

    # reshape the 2D cloud mask into a 1D vector
    vec_mask = cloud_mask.reshape(im.shape[0] * im.shape[1])

    # if image contains several bands, stretch the contrast for each band
    if len(im.shape) > 2:
        # reshape into a vector
        vec =  im.reshape(im.shape[0] * im.shape[1], im.shape[2])
        # initiliase with NaN values
        vec_adj = np.ones((len(vec_mask), im.shape[2])) * np.nan
        # loop through the bands
        for i in range(im.shape[2]):
            # find the higher percentile (based on prob)
            prc_high = np.percentile(vec[~vec_mask, i], prob_high)
            # clip the image around the 2 percentiles and rescale the contrast
            vec_rescaled = exposure.rescale_intensity(vec[~vec_mask, i],
                                                      in_range=(prc_low, prc_high))
            vec_adj[~vec_mask,i] = vec_rescaled
        # reshape into image
        im_adj = vec_adj.reshape(im.shape[0], im.shape[1], im.shape[2])

    # if image only has 1 bands (grayscale image)
    else:
        vec =  im.reshape(im.shape[0] * im.shape[1])
        vec_adj = np.ones(len(vec_mask)) * np.nan
        prc_high = np.percentile(vec[~vec_mask], prob_high)
        vec_rescaled = exposure.rescale_intensity(vec[~vec_mask], in_range=(prc_low, prc_high))
        vec_adj[~vec_mask] = vec_rescaled
        im_adj = vec_adj.reshape(im.shape[0], im.shape[1])

    return im_adj

def preprocess_single(fn, satname, cloud_mask_issue):
    """
    Reads the image and outputs the pansharpened/down-sampled multispectral bands,
    the georeferencing vector of the image (coordinates of the upper left pixel),
    the cloud mask, the QA band and a no_data image. 
    For Landsat 7-8 it also outputs the panchromatic band and for Sentinel-2 it
    also outputs the 20m SWIR band.

    KV WRL 2018

    Mark Lundine edit: get rid of extra image outputs, just rgb+ir bands (we get rid of ir later)
    Arguments:
    -----------
    fn: str or list of str
        filename of the .TIF file containing the image. For L7, L8 and S2 this 
        is a list of filenames, one filename for each band at different
        resolution (30m and 15m for Landsat 7-8, 10m, 20m, 60m for Sentinel-2)
    satname: str
        name of the satellite mission (e.g., 'L5')
    cloud_mask_issue: boolean
        True if there is an issue with the cloud mask and sand pixels are being masked on the images

    Returns:
    -----------
    im_ms: np.array
        3D array containing the pansharpened/down-sampled bands (B,G,R,NIR,SWIR1)
    georef: np.array
        vector of 6 elements [Xtr, Xscale, Xshear, Ytr, Yshear, Yscale] defining the
        coordinates of the top-left pixel of the image
    cloud_mask: np.array
        2D cloud mask with True where cloud pixels are
    im_extra : np.array
        2D array containing the 20m resolution SWIR band for Sentinel-2 and the 15m resolution
        panchromatic band for Landsat 7 and Landsat 8. This field is empty for Landsat 5.
    im_QA: np.array
        2D array containing the QA band, from which the cloud_mask can be computed.
    im_nodata: np.array
        2D array with True where no data values (-inf) are located

    """

    #=============================================================================================#
    # L5 images
    #=============================================================================================#
    if satname == 'L5':

        # read all bands
        data = gdal.Open(fn, gdal.GA_ReadOnly)
        georef = np.array(data.GetGeoTransform())
        bands = [data.GetRasterBand(k + 1).ReadAsArray() for k in range(data.RasterCount)]
        im_ms = np.stack(bands, 2)

        # down-sample to 15 m (half of the original pixel size)
        nrows = im_ms.shape[0]*2
        ncols = im_ms.shape[1]*2

        # create cloud mask
        im_QA = im_ms[:,:,5]
        im_ms = im_ms[:,:,:-1]
        cloud_mask = create_cloud_mask(im_QA, satname, cloud_mask_issue)

        # resize the image using bilinear interpolation (order 1)
        im_ms = transform.resize(im_ms,(nrows, ncols), order=1, preserve_range=True,
                                 mode='constant')
        # resize the image using nearest neighbour interpolation (order 0)
        cloud_mask = transform.resize(cloud_mask, (nrows, ncols), order=0, preserve_range=True,
                                      mode='constant').astype('bool_')

        # adjust georeferencing vector to the new image size
        # scale becomes 15m and the origin is adjusted to the center of new top left pixel
        georef[1] = 15
        georef[5] = -15
        georef[0] = georef[0] + 7.5
        georef[3] = georef[3] - 7.5
        
        # check if -inf or nan values on any band and add to cloud mask
        im_nodata = np.zeros(cloud_mask.shape).astype(bool)
        for k in range(im_ms.shape[2]):
            im_inf = np.isin(im_ms[:,:,k], -np.inf)
            im_nan = np.isnan(im_ms[:,:,k])
            cloud_mask = np.logical_or(np.logical_or(cloud_mask, im_inf), im_nan)
            im_nodata = np.logical_or(np.logical_or(im_nodata, im_inf), im_nan)
        # check if there are pixels with 0 intensity in the Green, NIR and SWIR bands and add those
        # to the cloud mask as otherwise they will cause errors when calculating the NDWI and MNDWI
        im_zeros = np.ones(cloud_mask.shape).astype(bool)
        for k in [1,3,4]: # loop through the Green, NIR and SWIR bands
            im_zeros = np.logical_and(np.isin(im_ms[:,:,k],0), im_zeros)
        # update cloud mask and nodata
        cloud_mask = np.logical_or(im_zeros, cloud_mask)
        im_nodata = np.logical_or(im_zeros, im_nodata)
        # no extra image for Landsat 5 (they are all 30 m bands)
        im_extra = []

    #=============================================================================================#
    # L7 images
    #=============================================================================================#
    elif satname == 'L7':

        # read pan image
        fn_pan = fn[0]
        data = gdal.Open(fn_pan, gdal.GA_ReadOnly)
        georef = np.array(data.GetGeoTransform())
        bands = [data.GetRasterBand(k + 1).ReadAsArray() for k in range(data.RasterCount)]
        im_pan = np.stack(bands, 2)[:,:,0]

        # size of pan image
        nrows = im_pan.shape[0]
        ncols = im_pan.shape[1]

        # read ms image
        fn_ms = fn[1]
        data = gdal.Open(fn_ms, gdal.GA_ReadOnly)
        bands = [data.GetRasterBand(k + 1).ReadAsArray() for k in range(data.RasterCount)]
        im_ms = np.stack(bands, 2)

        # create cloud mask
        im_QA = im_ms[:,:,5]
        cloud_mask = create_cloud_mask(im_QA, satname, cloud_mask_issue)

        # resize the image using bilinear interpolation (order 1)
        im_ms = im_ms[:,:,:5]
        im_ms = transform.resize(im_ms,(nrows, ncols), order=1, preserve_range=True,
                                 mode='constant')
        # resize the image using nearest neighbour interpolation (order 0)
        cloud_mask = transform.resize(cloud_mask, (nrows, ncols), order=0, preserve_range=True,
                                      mode='constant').astype('bool_')
        # check if -inf or nan values on any band and eventually add those pixels to cloud mask
        im_nodata = np.zeros(cloud_mask.shape).astype(bool)
        for k in range(im_ms.shape[2]):
            im_inf = np.isin(im_ms[:,:,k], -np.inf)
            im_nan = np.isnan(im_ms[:,:,k])
            cloud_mask = np.logical_or(np.logical_or(cloud_mask, im_inf), im_nan)
            im_nodata = np.logical_or(np.logical_or(im_nodata, im_inf), im_nan)
        # check if there are pixels with 0 intensity in the Green, NIR and SWIR bands and add those
        # to the cloud mask as otherwise they will cause errors when calculating the NDWI and MNDWI
        im_zeros = np.ones(cloud_mask.shape).astype(bool)
        for k in [1,3,4]: # loop through the Green, NIR and SWIR bands
            im_zeros = np.logical_and(np.isin(im_ms[:,:,k],0), im_zeros)
        # update cloud mask and nodata
        cloud_mask = np.logical_or(im_zeros, cloud_mask)
        im_nodata = np.logical_or(im_zeros, im_nodata)

        # pansharpen Green, Red, NIR (where there is overlapping with pan band in L7)
        try:
            im_ms_ps = pansharpen(im_ms[:,:,[1,2,3]], im_pan, cloud_mask)
        except: # if pansharpening fails, keep downsampled bands (for long runs)
            im_ms_ps = im_ms[:,:,[1,2,3]]
        # add downsampled Blue and SWIR1 bands
        im_ms_ps = np.append(im_ms[:,:,[0]], im_ms_ps, axis=2)
        im_ms_ps = np.append(im_ms_ps, im_ms[:,:,[4]], axis=2)

        im_ms = im_ms_ps.copy()
        # the extra image is the 15m panchromatic band
        im_extra = im_pan

    #=============================================================================================#
    # L8 images
    #=============================================================================================#
    elif satname == 'L8':

        # read pan image
        fn_pan = fn[0]
        data = gdal.Open(fn_pan, gdal.GA_ReadOnly)
        georef = np.array(data.GetGeoTransform())
        bands = [data.GetRasterBand(k + 1).ReadAsArray() for k in range(data.RasterCount)]
        im_pan = np.stack(bands, 2)[:,:,0]

        # size of pan image
        nrows = im_pan.shape[0]
        ncols = im_pan.shape[1]

        # read ms image
        fn_ms = fn[1]
        data = gdal.Open(fn_ms, gdal.GA_ReadOnly)
        bands = [data.GetRasterBand(k + 1).ReadAsArray() for k in range(data.RasterCount)]
        im_ms = np.stack(bands, 2)

        # create cloud mask
        im_QA = im_ms[:,:,5]
        cloud_mask = create_cloud_mask(im_QA, satname, cloud_mask_issue)

        # resize the image using bilinear interpolation (order 1)
        im_ms = im_ms[:,:,:5]
        im_ms = transform.resize(im_ms,(nrows, ncols), order=1, preserve_range=True,
                                 mode='constant')
        # resize the image using nearest neighbour interpolation (order 0)
        cloud_mask = transform.resize(cloud_mask, (nrows, ncols), order=0, preserve_range=True,
                                      mode='constant').astype('bool_')
        # check if -inf or nan values on any band and eventually add those pixels to cloud mask
        im_nodata = np.zeros(cloud_mask.shape).astype(bool)
        for k in range(im_ms.shape[2]):
            im_inf = np.isin(im_ms[:,:,k], -np.inf)
            im_nan = np.isnan(im_ms[:,:,k])
            cloud_mask = np.logical_or(np.logical_or(cloud_mask, im_inf), im_nan)
            im_nodata = np.logical_or(np.logical_or(im_nodata, im_inf), im_nan)
        # check if there are pixels with 0 intensity in the Green, NIR and SWIR bands and add those
        # to the cloud mask as otherwise they will cause errors when calculating the NDWI and MNDWI
        im_zeros = np.ones(cloud_mask.shape).astype(bool)
        for k in [1,3,4]: # loop through the Green, NIR and SWIR bands
            im_zeros = np.logical_and(np.isin(im_ms[:,:,k],0), im_zeros)
        # update cloud mask and nodata
        cloud_mask = np.logical_or(im_zeros, cloud_mask)
        im_nodata = np.logical_or(im_zeros, im_nodata)

        # pansharpen Blue, Green, Red (where there is overlapping with pan band in L8)
        try:
            im_ms_ps = pansharpen(im_ms[:,:,[0,1,2]], im_pan, cloud_mask)
        except: # if pansharpening fails, keep downsampled bands (for long runs)
            im_ms_ps = im_ms[:,:,[0,1,2]]
        # add downsampled NIR and SWIR1 bands
        im_ms_ps = np.append(im_ms_ps, im_ms[:,:,[3,4]], axis=2)

        im_ms = im_ms_ps.copy()
        # the extra image is the 15m panchromatic band
        im_extra = im_pan



    #=============================================================================================#
    # S2 images
    #=============================================================================================#
    if satname == 'S2':

        # read 10m bands (R,G,B,NIR)
        fn10 = fn[0]
        data = gdal.Open(fn10, gdal.GA_ReadOnly)
        georef = np.array(data.GetGeoTransform())
        bands = [data.GetRasterBand(k + 1).ReadAsArray() for k in range(data.RasterCount)]
        im10 = np.stack(bands, 2)
        im10 = im10/10000 # TOA scaled to 10000

        # if image contains only zeros (can happen with S2), skip the image
        if sum(sum(sum(im10))) < 1:
            im_ms = []
            georef = []
            # skip the image by giving it a full cloud_mask
            cloud_mask = np.ones((im10.shape[0],im10.shape[1])).astype('bool')
            return im_ms, georef, cloud_mask

        # size of 10m bands
        nrows = im10.shape[0]
        ncols = im10.shape[1]

        # read 20m band (SWIR1)
        fn20 = fn[1]
        data = gdal.Open(fn20, gdal.GA_ReadOnly)
        bands = [data.GetRasterBand(k + 1).ReadAsArray() for k in range(data.RasterCount)]
        im20 = np.stack(bands, 2)
        im20 = im20[:,:,0]
        im20 = im20/10000 # TOA scaled to 10000

        # resize the image using bilinear interpolation (order 1)
        im_swir = transform.resize(im20, (nrows, ncols), order=1, preserve_range=True,
                                   mode='constant')
        im_swir = np.expand_dims(im_swir, axis=2)

        # append down-sampled SWIR1 band to the other 10m bands
        im_ms = np.append(im10, im_swir, axis=2)

        # create cloud mask using 60m QA band (not as good as Landsat cloud cover)
        fn60 = fn[2]
        data = gdal.Open(fn60, gdal.GA_ReadOnly)
        bands = [data.GetRasterBand(k + 1).ReadAsArray() for k in range(data.RasterCount)]
        im60 = np.stack(bands, 2)
        im_QA = im60[:,:,0]
        cloud_mask = create_cloud_mask(im_QA, satname, cloud_mask_issue)
        # resize the cloud mask using nearest neighbour interpolation (order 0)
        cloud_mask = transform.resize(cloud_mask,(nrows, ncols), order=0, preserve_range=True,
                                      mode='constant')
        # check if -inf or nan values on any band and add to cloud mask
        im_nodata = np.zeros(cloud_mask.shape).astype(bool)
        for k in range(im_ms.shape[2]):
            im_inf = np.isin(im_ms[:,:,k], -np.inf)
            im_nan = np.isnan(im_ms[:,:,k])
            cloud_mask = np.logical_or(np.logical_or(cloud_mask, im_inf), im_nan)
            im_nodata = np.logical_or(np.logical_or(im_nodata, im_inf), im_nan)

        # check if there are pixels with 0 intensity in the Green, NIR and SWIR bands and add those
        # to the cloud mask as otherwise they will cause errors when calculating the NDWI and MNDWI
        im_zeros = np.ones(cloud_mask.shape).astype(bool)
        for k in [1,3,4]: # loop through the Green, NIR and SWIR bands
            im_zeros = np.logical_and(np.isin(im_ms[:,:,k],0), im_zeros)
        # update cloud mask and nodata
        cloud_mask = np.logical_or(im_zeros, cloud_mask)
        im_nodata = np.logical_or(im_zeros, im_nodata)

    return im_ms, georef, cloud_mask


def create_jpg(im_ms, cloud_mask, date, satname, sitename, filepath):
    """
    Saves a .jpg file with the RGB image as well as the NIR and SWIR1 grayscale images.
    This functions can be modified to obtain different visualisations of the 
    multispectral images.

    KV WRL 2018

    Arguments:
    -----------
    im_ms: np.array
        3D array containing the pansharpened/down-sampled bands (B,G,R,NIR,SWIR1)
    cloud_mask: np.array
        2D cloud mask with True where cloud pixels are
    date: str
        string containing the date at which the image was acquired
    satname: str
        name of the satellite mission (e.g., 'L5')

    Returns:
    -----------
        Saves a .jpg image corresponding to the preprocessed satellite image

    """

    im_RGB = rescale_image_intensity(im_ms[:,:,[2,1,0]], cloud_mask, 99.9)
    
    im_RGB = (im_RGB * 255).astype('uint8')
 
    cv2.imwrite(os.path.join(filepath,date + '_' + satname + sitename + '.jpg'), cv2.cvtColor(im_RGB, cv2.COLOR_RGB2BGR))


def save_jpg(metadata, settings, **kwargs):
    """
    Saves a .jpg image for all the images contained in metadata.

    KV WRL 2018

    Arguments:
    -----------
    metadata: dict
        contains all the information about the satellite images that were downloaded
    settings: dict with the following keys
        'inputs': dict
            input parameters (sitename, filepath, polygon, dates, sat_list)
        'cloud_thresh': float
            value between 0 and 1 indicating the maximum cloud fraction in 
            the cropped image that is accepted
        'cloud_mask_issue': boolean
            True if there is an issue with the cloud mask and sand pixels
            are erroneously being masked on the images
            
    Returns:
    -----------
    Stores the images as .jpg in a folder named /preprocessed
    
    """
    
    sitename = settings['inputs']['sitename']
    cloud_thresh = settings['cloud_thresh']
    filepath_data = settings['inputs']['filepath']

    # create subfolder to store the jpg files
    filepath_jpg = os.path.join(filepath_data, sitename, 'jpg_files', 'preprocessed')
    if not os.path.exists(filepath_jpg):
            os.makedirs(filepath_jpg)

    # loop through satellite list
    for satname in metadata.keys():

        filepath = SDS_tools.get_filepath(settings['inputs'],satname)
        filenames = metadata[satname]['filenames']

        # loop through images
        for i in range(len(filenames)):
            # image filename
            fn = SDS_tools.get_filenames(filenames[i],filepath, satname)
            # read and preprocess image
            im_ms, georef, cloud_mask = preprocess_single(fn, satname, False)
            # calculate cloud cover
            cloud_cover = np.divide(sum(sum(cloud_mask.astype(int))),
                                    (cloud_mask.shape[0]*cloud_mask.shape[1]))
            # skip image if cloud cover is above threshold
            if cloud_cover > cloud_thresh or cloud_cover == 1:
                continue
            # save .jpg with date and satellite in the title
            date = filenames[i][:19]
            create_jpg(im_ms, cloud_mask, date, satname, sitename, filepath_jpg)

    # print the location where the images have been saved
    print('Satellite images saved as .jpg in ' + os.path.join(filepath_data, sitename,
                                                    'jpg_files', 'preprocessed'))


