# Shoreline_Extraction_GAN

A work in progess in its beginning stages...more code and detail to come.

Generating binary land vs. water masks from coastal RGB imagery using [pix2pix](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix) and [cycle-GAN](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix).

This repo will (eventually) contain the following:
1. Code to help with setting up the training data for either of these image-to-image translation GANs.
2. Code to train both models.
3. Code to run the models.
4. Code to process the results and extract shorelines into GIS format.
5. Trained models for Delmarva's coast.
6. Code to download new imagery to run the model on.

# Examples

![cape_ex](/images/capehenlopen_ex.png)

![iri_ex](/images/iri_example.png)

![cape_map](/images/capeHenlopen_length_years.png)

![more_examples](/images/input_output_shoreline.png)

# Anaconda Environment Setup

envs/ holds two environment files that list all of the requirements.

pix2pix_shoreline is used for running the GANs, while shoreline_gan is for everything else.

Use Anaconda to set these up.
    
    conda env create --file envs/pix2pix_shoreline.yml
    conda env create --file envs/shoreline_gan.yml

# Workflow

shoreline_extraction.py contains the main modules for this project. The modules are described below.
Activate shoreline_gan before running modules.

    conda activate shoreline_gan

# Downloading satellite data

Data for this project was downloaded using [CoastSat](https://github.com/kvos/CoastSat).
CoastSat allows users to download Landsat 5, 7, 8 and Sentinel-2 imagery from anywhere around the world.
CoastSat also includes a number of useful preprocessing tools.

    download_imagery(polygon, dates, sat_list, sitename):
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
    """

This function will create a folder called 'data', and then a subdirectory for the site. 
In the site folder, there will be subdirectories for each satellite.
These subdirectories will contain all of the geotiff imagery.
These images go through the CoastSat preprocessing tools to make RGB jpegs.
These jpegs get saved to data/sitename/preprocessed/jpgs.
The jpegs are what get fed to the GAN during training and deployment.

# Compiling metadata

    get_metadata(site_folder, save_file):
    """
    gets metadata from geotiffs downloaded thru coastsat

    inputs:
    site_folder: path to site (str)
    save_file: path to csv to save metadata to (str)
    outputs:
    num_images: number of images (int)
    metadata_csv: path to the metadata csv (str)
    """
This function compiles the metadata for each geotiff downloaded through the CoastSat tools.
The csv will contain the image name, corner coordinates, x/y resolution, and epsg code for each image.
You feed it the filepath to the site containing all of the CoastSat downloaded geotiffs, as well as a csv path to save the metadata to.

# Labelling data.

I used [labelme](https://github.com/wkentaro/labelme) to manually segment imagery into two classes: land and water.
This image labeller saves the annotation files to jsons. To train the GANs, we need pngs.

# Converting jsons to pngs, setting up data to train a pix2pix model

Labelme comes with code that can convert jsons to pngs. I use that code to do this conversion.
I also made some code to set up training data for a pix2pix model.

    json_to_png(annotation_folder, extra_outputs, pix2pix_training_folder):
    """
    needs labelme installed and activated
    this converts labelme jsons to pngs
    sets up training, testing, and validation sets for a pix2pix model

    inputs: 
    annotation_folder: path to the folder containing the annotations (str)
    extra_outputs: path to save the conversions to (str)
    pix2pix_training_folder: path to save the pix2pix training set to (str)
    """

# Splitting Images

This step is important. The generators used in pix2pix and cycle-GAN require images of specific sizes. 
Satellite imagery is going to come in a variety of shapes and sizes.
The one I used (unet-256), expects images that are (256x256) in pixels.
So I have some code that basically splits every image into two overlapping square images.
Then each resulting image gets resized to 256x256.

    split_and_resize(image_folder,new_image_folder):
    """
    splits input images into two square images
    then resizes so they both have width/height of 256
    inputs:
    image_folder: path to input jpegs (str)
    output_folder: path to output jpegs (str)
    """

# Training/Testing/Validation Data Preparation

This code is used to combine A and B datasets for pix2pix training.

    setup_datasets(home,
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

Two examples of combined AB images are shown below.

![herringpointab](/images/herringptab.jpeg)
![fenwickab](/images/fenwickab.jpeg)

# Training

This module can be used to train a pix2pix or cycle-GAN model, after training/testing/validation data is prepped and organized.

    train_model(model_name,
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

# Deployment

Running the pix2pix model or cycle-GAN model will generate binary images from RGB L5, L7, L8, or S2 jpegs.
These images are then processed with the marching squares algorithm to extract contour lines for the shore.
These contours are then converted to the correct geographic coordinate system.
    
    run_and_process(site,
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
