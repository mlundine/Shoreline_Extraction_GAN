# Mark Lundine

import cv2
import numpy as np
import glob
import matplotlib.pyplot as plt
import os
import gdal_functions_app as gda
from osgeo import ogr
from osgeo import osr
import pandas as pd
import warnings
from shutil import copyfile
from skimage import measure
import geopandas as gpd
import shapely
warnings.filterwarnings("ignore")

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

def LineString_to_arr(line):
    listarray = []
    for pp in line.coords:
        listarray.append(pp)
    nparray = np.array(listarray)
    return nparray

def simplify_and_smooth_arr(contour, name):
    line = arr_to_LineString(contour)
    
    idx_L5 = name.find('L5')
    idx_L7 = name.find('L7')
    idx_L8 = name.find('L8')
    idx_S2 = name.find('S2')

    if idx_S2 > 0:
        simplify_tolerance = 10
    else:
        simplify_tolerance = 30
    
    simple = line.simplify(tolerance=1)
    simple_arr = LineString_to_arr(simple)
    simple_smooth_arr = chaikins_corner_cutting(simple_arr)
    return simple_smooth_arr

def simplify_lines(shapefile, tolerance=40):
    """
    Uses shapely simplify function to smooth out the extracted shorelines
    inputs:
    shapefile: path to merged shoreline shapefiles
    tolerance (optional): simplification tolerance (meters)
    outputs:
    save_path: path to smooth shapefile
    """

    save_path = os.path.splitext(shapefile)[0]+'simplify'+str(tolerance)+'.shp'
    lines = gpd.read_file(shapefile)
    lines['geometry'] = lines['geometry'].simplify(tolerance)
    lines.to_file(save_path)
    return save_path

def simplify_and_smooth(shapefile):

    name = os.path.basename(shapefile)
    dirname = os.path.dirname(shapefile)
    dirname = os.path.dirname(dirname)
    
    idx_L5 = name.find('L5')
    idx_L7 = name.find('L7')
    idx_L8 = name.find('L8')
    idx_S2 = name.find('S2')

    if idx_S2 > 0:
        tolerance = 10
    else:
        tolerance = 30


    
    simple_folder = os.path.join(dirname, 'simple')
    try:
        os.mkdir(simple_folder)
    except:
        pass
    simple_save_path = os.path.join(simple_folder, os.path.splitext(name)[0]+'simplify'+str(tolerance)+'.shp')
    lines = gpd.read_file(shapefile)
    lines['geometry'] = lines['geometry'].simplify(tolerance)
    lines.to_file(simple_save_path)

    simple_smooth = smooth_lines(simple_save_path)
    
    return simple_smooth

def vertex_filter(shapefile):
    gdf = gpd.read_file(shapefile)
    
    count = len(gdf)
    new_count = None
    for index, row in gdf.iterrows():
        gdf.at[index,'vtx'] = len(row['geometry'].coords)
    filter_gdf = gdf.copy()


    
    while count != new_count:
        count = len(filter_gdf)
        sigma = np.std(filter_gdf['vtx'])
        mean = np.mean(filter_gdf['vtx'])
        limit = mean+3*sigma
        filter_gdf = gdf[gdf['vtx']< limit]
        if mean < 5:
            break
        new_count = len(filter_gdf)
    
    new_path = os.path.splitext(shapefile)[0]+'vtx.shp'
    filter_gdf.to_file(new_path)
    return new_path


def chaikins_corner_cutting(coords, refinements=5):
    i=0
    for _ in range(refinements):
        L = coords.repeat(2, axis=0)
        R = np.empty_like(L)
        R[0] = L[0]
        R[2::2] = L[1:-1:2]
        R[1:-1:2] = L[2::2]
        R[-1] = L[-1]
        coords = L * 0.75 + R * 0.25
        i=i+1
    return coords

def smooth_lines(shapefile):
    dirname = os.path.dirname(shapefile)
    dirname = os.path.dirname(dirname)
    smooth_folder = os.path.join(dirname, 'smooth')
    try:
        os.mkdir(smooth_folder)
    except:
        pass
    save_path = os.path.join(smooth_folder,os.path.splitext(os.path.basename(shapefile))[0]+'smooth.shp')
    lines = gpd.read_file(shapefile)
    new_lines = lines.copy()
    for i in range(len(lines)):
        line = lines.iloc[i]
        coords = np.array(line.geometry)
        refined = chaikins_corner_cutting(coords)
        refined_geom = arr_to_LineString(refined)
        new_lines['geometry'][i] = refined_geom
    new_lines.to_file(save_path)
    return save_path
    
def kml_line(input_path, output_path):
    """
    Converts shapefile to kml
    inputs:
    input_path: path to input shapefile
    output_path: path to output kml
    """
    cmd = 'ogr2ogr -f ' + '"'+'KML' + '" ' + output_path + ' ' + input_path
    os.system(cmd)
    
def myLine(coords):
    """
    Makes a line feature from a list of xy tuples
    inputs: coords
    outputs: line
    """
    line = ogr.Geometry(type=ogr.wkbLineString)
    for xy in coords:
        line.AddPoint_2D(float(xy[0]),float(xy[1]))
    return line

def writePolyLineShp(line,
                     save_path,
                     epsg):
    """
    writes shapefile for extracted shoreline
    """

    # get timestamp and year from image name
    timestamp = os.path.basename(save_path)[0:19]
    year = int(timestamp[0:4])

    
    # create the shapefile
    driver = ogr.GetDriverByName("Esri Shapefile")
    ds = driver.CreateDataSource(save_path)
    
    # create the spatial reference system, WGS84
    srs =  osr.SpatialReference()
    srs.ImportFromEPSG(epsg)
    layr1 = ds.CreateLayer('line',srs, ogr.wkbLineString)

    # create the field
    layr1.CreateField(ogr.FieldDefn('timestamp', ogr.OFTString))
    layr1.CreateField(ogr.FieldDefn('year',ogr.OFTInteger))

    # Create the features and set values
    defn = layr1.GetLayerDefn()
    feat = ogr.Feature(defn)
    feat.SetField('timestamp', timestamp)
    feat.SetField('year', year)
    feat.SetGeometry(line)
    layr1.CreateFeature(feat)

    # close the shapefile
    ds.Destroy()
    
    return save_path

def translate_to_geo(points, geo_info, image):
    """
    translates local coordinates to geocoordinates
    """
    xmin,xmax,ymin,ymax,xres,yres = geo_info
    cols = int(xmax-xmin)/xres
    rows = int(ymax-ymin)/yres
    geo_points = np.zeros(np.shape(points))
    for i in range(len(points)):
        if image.find('one_fake')>0:
            if rows>cols:
                new_res_y = (cols/256)*yres
                new_res_x = (cols/256)*xres
                y,x = points[i]
                x_geo = xmin + (x*new_res_x)
                y_geo = ymax - (y*new_res_y)
                geo_points[i] = (x_geo,y_geo)
            else:
                new_res_y = (rows/256)*yres
                new_res_x = (rows/256)*xres
                y,x  = points[i]
                x_geo = xmin + (x*new_res_x)
                y_geo = ymax - (y*new_res_y)
                geo_points[i] = (x_geo,y_geo)
        else:
            if rows>cols:
                new_res_y = (cols/256)*yres
                new_res_x = (cols/256)*xres
                diff = rows-cols
                y,x  = points[i]
                x_geo = xmin + x*new_res_x
                y_geo = ymax - (diff*yres)
                y_geo = y_geo - (y*new_res_y)
                geo_points[i] = (x_geo,y_geo)
            else:
                new_res_y = (rows/256)*yres
                new_res_x = (rows/256)*xres
                diff = cols-rows
                y,x  = points[i]
                x_geo = xmin + (diff*xres)
                x_geo = x_geo + (x*new_res_x)
                y_geo = ymax - (y*new_res_y)
                geo_points[i] = (x_geo,y_geo)
    return geo_points

def get_geo_info(image, coords_file):
    """
    get image metadata
    inputs:
    image: path to an image
    coords_file: path to metadata csv
    outputs:
    [corner coordinates, resolution], epsg_code
    """
    image=os.path.basename(image)
    if image.find('one')>0:
        idx = image.find('one_real')
    else:
        idx = image.find('two_real')
    image = image[0:idx]
    df = pd.read_csv(coords_file)
    try:
        filtered = df[df['file']==image]
        filtered.reset_index()
        xmin = np.array(filtered['xmin'])[0]
        xmax = np.array(filtered['xmax'])[0]
        ymin = np.array(filtered['ymin'])[0]
        ymax = np.array(filtered['ymax'])[0]
        xres = np.array(filtered['xres'])[0]
        yres = np.array(filtered['yres'])[0]
        epsg = np.array(filtered['epsg'])[0]
    except:
        pass
    return [xmin,xmax,ymin,ymax,xres,yres],epsg

def filter_with_reference_shoreline(reference_shoreline, model_shorelines, distance_threshold=250):
    save_path = os.path.splitext(model_shorelines)[0]+'_ref_shoreline_filter.shp'


    reference_shp = gpd.read_file(reference_shoreline)
    buffer = reference_shp.buffer(distance_threshold,resolution=1)
    model_shp = gpd.read_file(model_shorelines)
    buffer_vals = [None]*len(model_shp)
    for i in range(len(model_shp)):
        line_entry = model_shp.iloc[i]
        line = line_entry.geometry
        bool_val = buffer.contains(line).values[0]
        buffer_vals[i] = bool_val
    model_shp['buffer_vals'] = buffer_vals
    model_shp_filter = model_shp[model_shp['buffer_vals']]
    model_shp_filter.to_file(save_path)

    return save_path

def filter_with_reference_region(reference_region_path, model_shorelines):
    save_path = os.path.splitext(model_shorelines)[0]+'_ref_region_filter.shp'
    reference_region = gpd.read_file(reference_region_path)
    model_shp = gpd.read_file(model_shorelines)
    buffer_vals = [None]*len(model_shp)
    for i in range(len(model_shp)):
        line_entry = model_shp.iloc[i]
        line = line_entry.geometry
        bool_val = reference_region.contains(line).values[0]
        buffer_vals[i] = bool_val
    model_shp['buffer_vals'] = buffer_vals
    model_shp_filter = model_shp[model_shp['buffer_vals']]
    model_shp_filter.to_file(save_path)

    return save_path
def extract_shorelines(pix2pix_outputs,
                       coords_file,
                       site_folder,
                       input_data,
                       clip_length=150):
    """
    Uses cv2.findContours to convert binary image from pix2pix to a shoreline feature class
    inputs:
    pix2pix_outputs: path to folder containing pix2pix generated images
    site_folder: path to site folder
    
    """

    ###get images into lists
    images = glob.glob(pix2pix_outputs+'\*.png')
    one_real = []
    one_rgb = []
    two_real = []
    two_rgb = []
    one_fake = []
    two_fake = []
    for file in images:
        if file.find('one_real')>0:
            one_real.append(file)
            name = os.path.basename(file)
            idx = name.find('_real')
            name = name[0:idx]+'.jpeg'
            one_rgb_im = os.path.join(input_data, name)
            one_rgb.append(one_rgb_im)
        elif file.find('two_real')>0:
            two_real.append(file)
            name = os.path.basename(file)
            idx = name.find('_real')
            name = name[0:idx]+'.jpeg'
            two_rgb_im = os.path.join(input_data, name)
            two_rgb.append(two_rgb_im)
        elif file.find('one_fake')>0:
            one_fake.append(file)
        else:
            two_fake.append(file)
    full = [one_real, one_fake, one_rgb, two_real, two_fake, two_rgb]
    num_images = len(full[0])

    ###loop over all images
    for i in range(num_images):
        one_real = full[0][i]
        one_fake = full[1][i]
        one_rgb = full[2][i]
        two_real = full[3][i]
        two_fake = full[4][i]
        two_rgb = full[5][i]

        ###open images
        one_fake_img = cv2.imread(one_fake)
        one_fake_img[one_fake_img>250] = 255
        one_fake_img[one_fake_img<255] = 0
        two_fake_img = cv2.imread(two_fake)
        two_fake_img[two_fake_img>250] = 255
        two_fake_img[two_fake_img<255] = 0

        ##convert to grayscale
        one_fake_img_gray = cv2.cvtColor(one_fake_img, cv2.COLOR_BGR2GRAY)
        rows_one,cols_one = np.shape(one_fake_img)[0:2]

        two_fake_img_gray = cv2.cvtColor(two_fake_img, cv2.COLOR_BGR2GRAY)
        rows_two,cols_two = np.shape(two_fake_img)[0:2]
        
        # apply binary thresholding, first image
        ret_one, thresh_one = cv2.threshold(one_fake_img_gray, 254, 255, cv2.THRESH_BINARY)

        # detect the contours on the binary image using marching squares
        contours_one = measure.find_contours(thresh_one, 254)
        
        # apply binary thresholding, second image
        ret_two, thresh_two = cv2.threshold(two_fake_img_gray, 254, 255, cv2.THRESH_BINARY)

        # detect the contours on the binary image using marching squares
        contours_two = measure.find_contours(thresh_two, 254)
        
        # get rid of short and extra contours
        contours_gis_one = contours_one
        maxlength=0
        i=0
        for contour in contours_gis_one:
            if len(contour)>maxlength:
                maxlength=len(contour)
                idx = i
            else:
                maxlength=maxlength
            i=i+1
        contour_one = contours_gis_one[idx]

        # get rid of short and extra contours
        contours_gis_two = contours_two
        maxlength=0
        i=0
        for contour in contours_gis_two:
            if len(contour)>maxlength:
                maxlength=len(contour)
                idx = i
            else:
                maxlength=maxlength
            i=i+1
        contour_two = contours_gis_two[idx]

        # save the results, image+shoreline overlay, shapefile
        name_one = os.path.splitext(os.path.basename(one_real))[0]
        name_two = os.path.splitext(os.path.basename(two_real))[0]
        idx = name_one.find('one')
        name = name_one[0:idx]
        
        #clipping ends
        idx_L5 = name.find('L5')
        idx_L7 = name.find('L7')
        idx_L8 = name.find('L8')
        idx_S2 = name.find('S2')

        if idx_S2 > 0:
            clip_units = int(clip_length/10)
        else:
            clip_units = int(clip_length/30)

        contour_one = contour_one[clip_units:-clip_units]
        contour_two = contour_two[clip_units:-clip_units]

        # get in utm coordinates
        geo_info_one,epsg_one = get_geo_info(one_real, coords_file)
        epsg_one=int(epsg_one)
        geo_points_one = translate_to_geo(contour_one, geo_info_one, one_fake)

        # get in utm coordinates
        geo_info_two,epsg_two = get_geo_info(two_real, coords_file)
        epsg_two=int(epsg_two)
        geo_points_two = translate_to_geo(contour_two, geo_info_two, two_fake)



        
        
        geo_points_one = geo_points_one
        geo_points_two = geo_points_two
        line1 = myLine(geo_points_one)
        line2 = myLine(geo_points_two)


        x_one = contour_one[:,0]
        y_one = contour_one[:,1]
        contour_one_nice = list(zip(x_one, y_one))
        contour_one_simple_smooth = simplify_and_smooth_arr(contour_one_nice, name_one)

        x_two = contour_two[:,0]
        y_two = contour_two[:,1]
        contour_two_nice = list(zip(x_two,y_two))
        contour_two_simple_smooth = simplify_and_smooth_arr(contour_two_nice, name_two)
            
        # saving shoreline overlay images
        shoreline_save = os.path.join(site_folder, 'shoreline_images')    
        name_im_one = os.path.join(shoreline_save, name_one+'overlayshore.png')
        name_im_two = os.path.join(shoreline_save, name_two+'overlayshore.png')   

        # draw contours on the original image
        one_real_copy = cv2.cvtColor(cv2.imread(one_rgb).copy(), cv2.COLOR_BGR2RGB)
        two_real_copy = cv2.cvtColor(cv2.imread(two_rgb).copy(), cv2.COLOR_BGR2RGB)
        
        # Display the image and plot all contours found
        fig, ax = plt.subplots()
        ax.imshow(one_real_copy,interpolation='nearest')

        ax.plot(contour_one_simple_smooth[:, 1], contour_one_simple_smooth[:, 0], linewidth=1,color='g')

        ax.axis('image')
        ax.set_xticks([])
        ax.set_yticks([])
        plt.savefig(name_im_one,bbox_inches='tight', dpi=300) #save image
        plt.close()
        
        # Display the image and plot all contours found
        fig, ax = plt.subplots()
        ax.imshow(two_real_copy,interpolation='nearest')

        ax.plot(contour_two_simple_smooth[:, 1], contour_two_simple_smooth[:, 0], linewidth=1,color='g')

        ax.axis('image')
        ax.set_xticks([])
        ax.set_yticks([])
        plt.savefig(name_im_two, bbox_inches='tight', dpi=300) #save image
        plt.close()

        # saving shapefile
        shapefile_save_one = os.path.join(site_folder, 'shapefiles', 'one')
        shapefile_save_one_raw = os.path.join(shapefile_save_one, 'raw')
        shapefile_save_two = os.path.join(site_folder, 'shapefiles', 'two')
        shapefile_save_two_raw = os.path.join(shapefile_save_two, 'raw')
        try:
            os.mkdir(shapefile_save_one)
        except:
            pass
        try:
            os.mkdir(shapefile_save_one_raw)
        except:
            pass
        try:
            os.mkdir(shapefile_save_two)
        except:
            pass
        try:
            os.mkdir(shapefile_save_two_raw)
        except:
            pass

        name_shape_1 = os.path.join(shapefile_save_one_raw, name_one+'shore.shp')
        name_shape_2 = os.path.join(shapefile_save_two_raw, name_two+'shore.shp')
        
        save_path1 = writePolyLineShp(line1, name_shape_1,epsg_one)
        save_path2 = writePolyLineShp(line2, name_shape_2,epsg_two)

        simplify_and_smooth(save_path1)
        simplify_and_smooth(save_path2)

def merge_shapefiles(shapefile_folder,
                     shapefile_merged,
                     val,
                     site):
    """
    Merges clipped shapefiles into one
    inputs:
    clipped_shapefile_folder: path to folder of clipped shapefiles (str)
    site: site name (str)
    outputs:
    merge_shape_path: the path to the output shapefile (str)
    """
    merge_shape_path = os.path.join(shapefile_merged, site+val+'.shp')
    gda.mergeShapes(shapefile_folder, merge_shape_path)
    return merge_shape_path


def process(pix2pix_outputs,
            site,
            coords_file,
            output_folder,
            input_data,
            reference_shoreline=None,
            reference_region=None,
            distance_threshold=250,
            clip_length=150):
    """
    Takes pix2pix outputs, extracts shorelines, outputs results in various formats
    inputs:
    pix2pix_outputs: folder to pix2pix images (str)
    site: site name (str)
    coords_file: path to metadata csv (str)
    output_folder: path to save outputs to (str)
    """

    ##Define output folders
    processed_folder = os.path.join(output_folder, 'processed')
    site_folder = os.path.join(processed_folder, site)
    shoreline_images = os.path.join(site_folder, 'shoreline_images')
    shapefile_folder = os.path.join(site_folder, 'shapefiles')
    shapefile_merged = os.path.join(site_folder, 'shapefile_merged')
    kml_folder = os.path.join(site_folder, 'kml_merged')
    output_folders = [site_folder, shoreline_images,
                      shapefile_folder, shapefile_merged, kml_folder]
    
    ##Make them if not already there
    for folder in output_folders:
        try:
            os.mkdir(folder)
        except:
            pass


    ##Extract shorelines from pix2pix outputs
    extract_shorelines(pix2pix_outputs,
                       coords_file,
                       site_folder,
                       input_data,
                       clip_length=clip_length)

    ##Merge shapefiles into one
    shapefile1 = merge_shapefiles(os.path.join(shapefile_folder, 'one', 'smooth'),
                                  shapefile_merged,
                                  'one',
                                  site)
    ##Merge shapefiles into one
    shapefile2 = merge_shapefiles(os.path.join(shapefile_folder, 'two', 'smooth'),
                                  shapefile_merged,
                                  'two',
                                  site)
    
    

    #Filters
    if reference_shoreline != None: 
        shore_1 = filter_with_reference_shoreline(reference_shoreline, shapefile1, distance_threshold=distance_threshold)
        shore_2 = filter_with_reference_shoreline(reference_shoreline, shapefile2, distance_threshold=distance_threshold)
        final_1 = vertex_filter(shore_1)
        final_2 = vertex_filter(shore_2)
        if reference_region != None:
            region_1 = filter_with_reference_region(reference_region, shapefile1)
            region_2 = filter_with_reference_region(reference_region, shapefile2)
            final_1 = vertex_filter(region_1)
            final_2 = vertex_filter(region_2)
    elif reference_region != None:
        region_1 = filter_with_reference_region(reference_region, shapefile1)
        region_2 = filter_with_reference_region(reference_region, shapefile2)
        final_1 = vertex_filter(region_1)
        final_2 = vertex_filter(region_2)
    else:
        final_1 = vertex_filter(shapefile1)
        final_2 = vertex_filter(shapefile2)
    
    ##Convert merged shapefile to kml
    kml_line(final_1, os.path.join(kml_folder, site+'_one_merged.kml'))
    kml_line(final_2, os.path.join(kml_folder, site+'_two_merged.kml'))
        














