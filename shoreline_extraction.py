# Mark Lundine

import cv2
import numpy as np
import glob
import matplotlib.pyplot as plt
import os
from gdal_modules import gdal_functions_app as gda
from osgeo import ogr
from osgeo import osr
import pandas as pd
import warnings
from shutil import copyfile
from skimage import measure
#from coastsat import SDS_download, SDS_preprocess2
import geopandas as gpd
warnings.filterwarnings("ignore")

def get_metadata(site_folder, save_file):
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
    filter_df.to_csv(metadata_csv, index=False)
    
    return num_images, metadata_csv


def json_to_png(annotation_folder, extra_outputs, pix2pix_training_folder):
    """
    needs labelme installed and activated
    this converts labelme jsons to pngs
    """
    
    folder = annotation_folder
    testA = os.path.join(pix2pix_training_folder, 'A', 'test')
    testB = os.path.join(pix2pix_training_folder, 'B', 'test')
    trainA = os.path.join(pix2pix_training_folder, 'A', 'train')
    trainB = os.path.join(pix2pix_training_folder, 'B', 'train')
    valA = os.path.join(pix2pix_training_folder, 'A', 'val')
    valB = os.path.join(pix2pix_training_folder, 'B', 'val')
    try:
        os.mkdir(extra_outputs)
        os.mkdir(testA)
        os.mkdir(testB)
        os.mkdir(trainA)
        os.mkdir(trainB)
        os.mkdir(valA)
        os.mkdir(valB)
    except:
        pass


    jsons = []
    for file in glob.glob(folder + '\*.json'):
        cmd = 'labelme_json_to_dataset ' + file + ' -o ' + os.path.join(extra_outputs, os.path.splitext(os.path.basename(file))[0])
        os.system(cmd)
        jsons.append(file)
    
    subfolders = [ f.path for f in os.scandir(extra_outputs) if f.is_dir() ] 
    for folder in subfolders:
        name = os.path.basename(folder)
        srcpng = os.path.join(folder, 'label.png')
        dstpng = os.path.join(extra_outputs, folder + '.png')
        copyfile(srcpng, dstpng)

    labels = []
    for file in glob.glob(extra_outputs + '\*.png'):
        labels.append(file)

    total = len(labels)
    i=0
    frac=0
    for lab in labels:
        name = os.path.splitext(os.path.basename(lab))[0]
        print(name)
        new_name = name+'.jpeg'
        new_lab = name+'.png'
        srcjpeg = os.path.join(folder, new_name)
        srcpng = lab
        if frac < 0.60:
            dstjpeg = os.path.join(trainA, new_name)
            dstpng = os.path.join(trainB, new_lab)
        elif frac < 0.80:
            dstjpeg = os.path.join(testA, new_name)
            dstpng = os.path.join(testB, new_lab)
        else:
            dstjpeg = os.path.join(valA, new_name)
            dstpng = os.path.join(valB, new_lab)
        copyfile(srcjpeg, dstjpeg)
        copyfile(srcpng, dstpng)
        i=i+1
        frac = i/total

        
def split_and_resize(image_folder,new_image_folder):
    """
    splits input images into two square images
    then resizes so they both have width/height of 256
    inputs:
    image_folder: path to input jpegs (str)
    output_folder: path to output jpegs (str)
    """
    for image in glob.glob(image_folder + '/*.jpg'):
        img = cv2.imread(image)
        rows,cols,bands = np.shape(img)
        if rows>cols:
            new_img = img[0:cols,0:cols]
            dif = rows-cols
            ###xmin is shifted to right by diff
            new_img2 = img[dif:rows,0:cols]
            nr,nc,nb = np.shape(new_img2)
            if nr!=nc:
                print(np.shape(new_img2))
        else:
            new_img = img[0:rows,0:rows]
            dif = cols-rows
            ###y is shifted down by diff, new ymin
            new_img2 = img[0:rows,dif:cols]
            nr,nc,nb = np.shape(new_img2)
            if nr!=nc:
                print(np.shape(new_img2))          
        newSize = (256,256)
        name = os.path.splitext(os.path.basename(image))[0]
        new_img = cv2.resize(new_img, newSize, interpolation = cv2.INTER_NEAREST)
        new_img2 = cv2.resize(new_img2, newSize, interpolation = cv2.INTER_NEAREST)
        new_image = os.path.join(new_image_folder, name+'one.jpeg')
        new_image2 = os.path.join(new_image_folder, name+'two.jpeg')
        cv2.imwrite(new_image, new_img)
        cv2.imwrite(new_image2, new_img2)


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


    Delmarva GAN Training Examples
    download_imagery([[-75.114065,38.813792],
         [-75.081164,38.814048],
         [-75.079118,38.789446],
         [-75.112575,38.790201]],
         ['1980-01-01','2022-01-18'],
         ['L7','L8','L5','S2'],
         'cape_henlopen')
    download_imagery([[-75.240862,38.838372],
         [-75.222254,38.845872],
         [-75.184636,38.813253],
         [-75.202656,38.803014]],
         ['1980-01-01','2022-01-18'],
         ['L5','L8', 'L7', 'S2'],
         'broadkill_beach')  
    main([[-75.094931,38.643827],
         [-75.047602,38.645155],
         [-75.038862,38.575201],
         [-75.090859,38.574462]],
         ['1980-01-01','2022-01-18'],
         ['L7','L8','L5','S2'],
         'indian_river_inlet')   
    main([[-75.095310, 38.781918],
         [-75.075011,38.782907],
         [-75.068781,38.749606],
         [-75.094453,38.748789]],
         ['1980-01-01','2022-01-18'],
         ['L7','L8','L5','S2'],
         'herring_point')   
    main([[-75.408723, 37.906953],
         [-75.323589,37.878013],
         [-75.363116,37.822948],
         [-75.444037,37.850573]],
         ['1980-01-01','2022-01-18'],
         ['L7','L8','L5','S2'],
         'fishing_point')  
    main([[-75.094453, 38.748789],
         [-75.068781,38.749606],
         [-75.057925, 38.705229],
         [-75.092661,38.702230]],
         ['1980-01-01','2022-01-18'],
         ['L7','L8','L5','S2'],
         'rehobothN')
    main([[-75.092661, 38.702230],
         [-75.057925, 38.705229],
         [-75.047602, 38.645155],
         [-75.094931, 38.643827]],
         ['1980-01-01','2022-01-18'],
         ['L7','L8','L5','S2'],
         'rehobothS','RGB',4)
    main([[-75.090859, 38.574462],
         [-75.038862, 38.575201],
         [-75.031354, 38.516628],
         [-75.085939,38.515177]],
         ['1980-01-01','2022-01-18'],
         ['L7','L8','L5','S2'],
         'bethany')
    main([[-75.085939,38.515177],
         [-75.031354, 38.516628],
         [-75.026715,38.450289],
         [-75.086682, 38.451093]],
         ['1980-01-01','2022-01-18'],
         ['L7','L8','L5','S2'],
         'fenwick')
    main([[-75.202656, 38.803014],
         [-75.184636, 38.813253],
         [-75.115804, 38.788154],
         [-75.134353, 38.769031]],
         ['1980-01-01','2022-01-18'],
         ['L7','L8','L5','S2'],
         'lewes')
    download_imagery([[-75.095456,  38.389022],
         [-75.032444, 38.387132],
         [-75.068989, 38.311750],
         [-75.112996,  38.31979]],
         ['1980-01-01','2022-01-18'],
         ['S2'],
         'oceancityS')
    download_imagery([[-75.086682, 38.451093],
         [-75.026715, 38.450289],
         [-75.032444, 38.387132],
         [-75.095456,  38.389022]],
         ['1980-01-01','2022-01-18'],
         ['L7','L8','L5','S2'],
         'oceancityN')
    download_imagery([[-75.202656, 38.803014],
         [-75.184636, 38.813253],
         [-75.115804, 38.788154],
         [-75.134353, 38.769031]],
         ['1980-01-01','2022-01-18'],
         ['L5','L7','L8'],
         'lewes_two')
    download_imagery([[-75.240862,38.838372],
         [-75.222254,38.845872],
         [-75.184636,38.813253],
         [-75.202656,38.803014]],
         ['1980-01-01','2022-01-18'],
         ['L5','L7', 'L8'],
         'broadkill_two')
    download_imagery([[-75.112996,  38.31979],
         [-75.068989, 38.311750],
         [-75.111311, 38.236194],
         [-75.153670,  38.244109]],
         ['1980-01-01','2022-01-18'],
         ['L7','L8','L5','S2'],
         'assateague1')
    download_imagery([[-75.153670,  38.244109],
         [-75.111311, 38.236194],
         [-75.147881, 38.156431],
         [-75.191571,  38.163960]],
         ['1980-01-01','2022-01-18'],
         ['L7','L8','L5','S2'],
         'assateague2')
    download_imagery([[-75.191571,  38.163960],
         [-75.147881, 38.156431],
         [-75.174407, 38.089413],
         [-75.216432,  38.096623]],
         ['1980-01-01','2022-01-18'],
         ['L7','L8','L5','S2'],
         'assateague3')
    download_imagery([[-75.216432,  38.096623],
         [-75.174407, 38.089413],
         [-75.214733, 38.024127],
         [-75.262969, 38.038174]],
         ['1980-01-01','2022-01-18'],
         ['L7','L8','L5','S2'],
         'assateague4')
    download_imagery([[-75.262969, 38.038174],
         [-75.214733, 38.024127],
         [-75.266823, 37.961199],
         [-75.305247,  37.978163]],
         ['1980-01-01','2022-01-18'],
         ['L7','L5'],
         'assateague5_two')
    download_imagery([[-75.305247,  37.978163],
         [-75.266823, 37.961199],
         [-75.319717, 37.876987],
         [-75.371751,  37.895427]],
         ['1980-01-01','2022-01-18'],
         ['L7','L8','L5'],
         'assateague6_two')
    download_imagery([[-75.411930, 37.895036],
         [-75.319402, 37.895646],
         [-75.346185, 37.834353],
         [-75.426657, 37.832667]],
         ['1980-01-01','2022-01-18'],
         ['L7','L8','L5','S2'],
         'fishingpoint')     
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
                'cloud_thresh': 0.5,        # threshold on maximum cloud cover
                'inputs':inputs
                }
    
    ### preprocess images (cloud masking, geotiff to jpeg conversion, write geo-metadata to jpeg)
    SDS_preprocess2.save_jpg(metadata, settings)

    ##saving metadata from geotiffs to csv
    site_folder = os.path.join(filepath_data, sitename)
    metadata_csv = get_metadata(site_folder,
                                os.path.join(site_folder, sitename+'.csv'))
    return metadata_csv, site_folder
    

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
        idx = image.find('one')
    else:
        idx = image.find('two')
    image = image[0:idx]
    df = pd.read_csv(coords_file)
    try:
        filtered = df[df['image']==image]
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


def extract_shorelines(pix2pix_outputs,
                       coords_file,
                       site_folder):
    """
    Uses cv2.findContours to convert binary image from pix2pix to a shoreline feature class
    inputs:
    pix2pix_outputs: path to folder containing pix2pix generated images
    site_folder: path to site folder
    
    """

    ###get images into lists
    images = glob.glob(pix2pix_outputs+'\*.png')
    one_real = []
    two_real = []
    one_fake = []
    two_fake = []
    for file in images:
        if file.find('one_real')>0:
            one_real.append(file)
        elif file.find('two_real')>0:
            two_real.append(file)
        elif file.find('one_fake')>0:
            one_fake.append(file)
        else:
            two_fake.append(file)
    full = [one_real, one_fake, two_real, two_fake]
    num_images = len(full[0])

    ###loop over all images
    for i in range(num_images):
        one_real = full[0][i]
        one_fake = full[1][i]
        two_real = full[2][i]
        two_fake = full[3][i]

        ###open images
        one_fake_img = cv2.imread(one_fake)
        one_fake_img[one_fake_img>254] = 255
        two_fake_img = cv2.imread(two_fake)
        two_fake_img[two_fake_img>254] = 255

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
        
        # get in utm coordinates
        geo_info_one,epsg_one = get_geo_info(one_real, coords_file)
        epsg_one=int(epsg_one)
        geo_points_one = translate_to_geo(contour_one, geo_info_one, one_fake)

        # get in utm coordinates
        geo_info_two,epsg_two = get_geo_info(two_real, coords_file)
        epsg_two=int(epsg_two)
        geo_points_two = translate_to_geo(contour_two, geo_info_two, two_fake)


        # save the results, image+shoreline overlay, shapefile
        name_one = os.path.splitext(os.path.basename(one_real))[0]
        name_two = os.path.splitext(os.path.basename(two_real))[0]
        idx = name_one.find('one')
        name = name_one[0:idx]
        
        if name.find('capehenlopen')<0:
            geo_points = np.concatenate((geo_points_one, geo_points_two))
            geo_points = sorted(geo_points , key=lambda k: [k[1], k[0]])
            ##make polyline
            line = myLine(geo_points)
        else:
            geo_points_one = geo_points_one
            geo_points_two = geo_points_two
            line = myLine(geo_points_two)
            
        # saving shoreline overlay images
        shoreline_save = os.path.join(site_folder, 'shoreline_images')    
        name_im_one = os.path.join(shoreline_save, name_one+'overlayshore.png')
        name_im_two = os.path.join(shoreline_save, name_two+'overlayshore.png')    

        # draw contours on the original image
        one_real_copy = cv2.imread(one_real).copy()
        two_real_copy = cv2.imread(two_real).copy()
        
        # Display the image and plot all contours found
        fig, ax = plt.subplots()
        ax.imshow(one_real_copy, cmap=plt.cm.gray)

        ax.plot(contour_one[:, 1], contour_one[:, 0], linewidth=1,color='g')

        ax.axis('image')
        ax.set_xticks([])
        ax.set_yticks([])
        plt.savefig(name_im_one, dpi=300) #save image
        plt.close()
        
        # Display the image and plot all contours found
        fig, ax = plt.subplots()
        ax.imshow(two_real_copy, cmap=plt.cm.gray)

        ax.plot(contour_two[:, 1], contour_two[:, 0], linewidth=1,color='g')

        ax.axis('image')
        ax.set_xticks([])
        ax.set_yticks([])
        plt.savefig(name_im_two, dpi=300) #save image
        plt.close()

        # saving shapefile
        shapefile_save = os.path.join(site_folder, 'shapefiles')
        name_shape = os.path.join(shapefile_save, name+'shore.shp')
        writePolyLineShp(line,name_shape,epsg_one)

def merge_shapefiles(shapefile_folder,
                     shapefile_merged,
                     site):
    """
    Merges clipped shapefiles into one
    inputs:
    clipped_shapefile_folder: path to folder of clipped shapefiles (str)
    site: site name (str)
    outputs:
    merge_shape_path: the path to the output shapefile (str)
    """
    merge_shape_path = os.path.join(shapefile_merged, site+'.shp')
    gda.mergeShapes(shapefile_folder, merge_shape_path)
    return merge_shape_path


def process(pix2pix_outputs,
            site,
            coords_file,
            output_folder):
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
                       site_folder)

    ##Merge shapefiles into one
    shapefile = merge_shapefiles(shapefile_folder,
                                 shapefile_merged,
                                 site)

    ##Convert merged shapefile to kml
    kml_line(shapefile, os.path.join(kml_folder, site+'merged.kml'))

def run_model(site,
              source,
              outputs_dir,
              num_images):
    """
    Runs trained pix2pix or cycle-GAN shoreline models
    inputs:
    site: name for site (str)
    source: folder with images to run on (str)
    outputs_dir: folder to save outputs to (str, ex : r'./outputs'
    num_images: number of images in source folder (int)
    outputs:
    save_folder: directory where generated images are saved (str)
    """
    root = os.getcwd()
    pix2pix_detect = os.path.join(root, 'pix2pix_modules', 'test.py')
    results_dir = os.path.join(outputs_dir, 'gan', site)
    try:
        os.mkdir(results_dir)
    except:
        pass
    cmd0 = 'conda deactivate & conda activate pix2pix_shoreline & '
    cmd1 = 'python ' + pix2pix_detect
    cmd2 = ' --dataroot ' + source
    cmd3 = ' --model test'
    cmd4 = ' --name Delmarva_shoreline' ##change this as input
    cmd5 = ' --netG unet_256'
    cmd6 = ' --netD basic'
    cmd7 = ' --dataset_mode single'
    cmd8 = ' --norm batch'
    cmd9 = ' --num_test ' + str(num_images)
    cmd10 = ' --preprocess none'
    cmd11 = ' --results_dir ' + results_dir
    cmd12 = ' --checkpoints_dir ' + os.path.join(root, 'pix2pix_modules', 'checkpoints')
    full_cmd = cmd0+cmd1+cmd2+cmd3+cmd4+cmd5+cmd6+cmd7+cmd8+cmd9+cmd10+cmd11+cmd12
    os.system(full_cmd)
    save_folder = os.path.join(results_dir,'Delmarva_shoreline','test_latest', 'images') ##change this is input
    return save_folder

def sort_images(site_dict, in_folder):
    jpgs = glob.glob(in_folder + '\*.jpg')
    jpegs = glob.glob(in_folder + '\*.jpeg')
    for site in site_dict:
        out_folder = site_dict[site]
        for im in jpgs:
            if im.find(site+'one')>0:
                src = im
                dst = os.path.join(out_folder, os.path.splitext(os.path.basename(im))[0]+'.jpeg')
                copyfile(src, dst)
            elif im.find(site+'two')>0:
                src = im
                dst = os.path.join(out_folder, os.path.splitext(os.path.basename(im))[0]+'.jpeg')
                copyfile(src, dst)
            else:
                continue
        for im in jpegs:
            if im.find(site+'one')>0:
                src = im
                dst = os.path.join(out_folder, os.path.splitext(os.path.basename(im))[0]+'.jpeg')
                copyfile(src, dst)
            elif im.find(site+'two')>0:
                src = im
                dst = os.path.join(out_folder, os.path.splitext(os.path.basename(im))[0]+'.jpeg')
                copyfile(src, dst)
            else:
                continue
            
def setup_datasets(home,
                   name,
                   foldA,
                   foldB):
    """
    Setups annotation pairs for pix2pix training
    inputs:
    home: parent directory for annotations (str) (ex: r'pix2pix_modules/datasets/MyProject')
    name: project name (str)
    foldA: path to A annotations (str)
    foldB: path to B annotations (str)
    """
    cmd0 = 'conda deactivate & conda activate pix2pix_shoreline & '
    cmd1 = 'python pixpix_modules/datasets/combine_A_and_B.py'
    cmd2 = ' --fold_A ' + foldA
    cmd3 = ' --fold_B ' + foldB
    cmd4 = ' --fold_AB ' + home
    cmd5 = ' --no_multiprocessing'
    full_cmd = cmd0+cmd1+cmd2+cmd3+cmd4+cmd5
    os.system(full_cmd)
    
def train_model(model_name,
                model_type,
                dataroot,
                n_epochs = 200):
    """
    Trains pix2pix or cycle-GAN model
    inputs:
    model_name: name for your model (str)
    model_type: either 'pix2pix' or 'cycle-GAN' (str)
    dataroot: path to training/test/val directories (str)
    n_epochs (optional): number of epochs to train for (int)
    """
    root = os.getcwd()
    pix2pix_train = os.path.join(root, 'pix2pix_modules', 'train.py')

    cmd0 = 'conda deactivate & conda activate pix2pix_shoreline & '
    cmd1 = 'python ' + pix2pix_train
    cmd2 = ' --dataroot ' + dataroot
    cmd3 = ' --model ' + model_type
    cmd4 = ' --name ' + model_name#change this as input
    cmd5 = ' --netG unet_256'
    cmd6 = ' --netD basic'
    cmd7 = ' --num_test ' + str(num_images)
    cmd8 = ' --preprocess none'
    cmd9 = ' --checkpoints_dir ' + os.path.join(root, 'pix2pix_modules', 'checkpoints')
    cmd10 = ' --n_epochs ' + str(n_epochs)
    full_cmd = cmd0+cmd1+cmd2+cmd3+cmd4+cmd5+cmd6+cmd7+cmd8+cmd9+cmd10
    os.system(full_cmd)
    print('Training Finished')


def run_and_process(site,
                    source,
                    coords_file):
    """
    Runs trained pix2pix or cycle-GAN model,
    then runs outputs through marching squares to extract shorelines,
    then converts the shorelines to GIS format.
    inputs:
    site: name for site (str)
    source: directory with images to run model on (str)
    coords_file: path to the csv containing metadata on images (str)
    """
    root = os.getcwd()
    outputs_dir = os.path.join(root, 'outputs')
    gan_dir = os.path.join(outputs_dir, 'gan')
    processed_dir = os.path.join(outputs_dir, 'processed')
    dirs = [outputs_dir, gan_dir, processed_dir]
    for dir in dirs:
        try:
            os.mkdir(dir)
        except:
            pass
    num_images = len(glob.glob(source+'\*.jpeg'))
    print('Running GAN')
    gan_results = run_model(site, source, outputs_dir, num_images)
    print('GAN finished')
    print('Extracting Shorelines')
    process(gan_results,
            site,
            coords_file,
            outputs_dir)
    print('Shorelines Extracted')
                     
##capehenlopen = r'D:\Shoreline_Extraction_GAN\data\delmarva\unlabelled_split\capehenlopen'
##fishingpoint = r'D:\Shoreline_Extraction_GAN\data\delmarva\unlabelled_split\fishingpoint'
##herringpoint = r'D:\Shoreline_Extraction_GAN\data\delmarva\unlabelled_split\herringpoint'
##iri = r'D:\Shoreline_Extraction_GAN\data\delmarva\unlabelled_split\iri'
##run_and_process('iri',
##                iri,
##                r'D:\Shoreline_Extraction_GAN\data\delmarva\metadata_delmarva_edit2.csv')



            
















