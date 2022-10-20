import os
import glob
from utils import shoreline_extraction_utils
    
def run_model(site,
              source,
              model_name,
              epoch,
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
    cmd4 = ' --name ' + model_name ##change this as input
    cmd5 = ' --netG unet_256'
    cmd6 = ' --netD basic'
    cmd7 = ' --dataset_mode single'
    cmd8 = ' --norm batch'
    cmd9 = ' --num_test ' + str(num_images)
    cmd10 = ' --preprocess none'
    cmd11 = ' --input_nc 3'
    cmd12 = ' --output_nc 1'
    cmd13 = ' --results_dir ' + results_dir
    cmd14 = ' --checkpoints_dir ' + os.path.join(root, 'pix2pix_modules', 'checkpoints')
    cmd15 = ' --epoch ' + epoch
    full_cmd = cmd0+cmd1+cmd2+cmd3+cmd4+cmd5+cmd6+cmd7+cmd8+cmd9+cmd10+cmd11+cmd13+cmd12+cmd13+cmd14+cmd15
    os.system(full_cmd)
    save_folder = os.path.join(results_dir,model_name, 'test_latest', 'images') ##change this is input
    return save_folder




def run_and_process(site,
                    source,
                    model_name,
                    coords_file,
                    epoch='latest'):
    """
    Runs trained pix2pix or cycle-GAN model,
    then runs outputs through marching squares to extract shorelines,
    then converts the shorelines to GIS format.
    inputs:
    site: name for site (str)
    source: directory with images to run model on (str)
    model_name: name for trained model (str)
    coords_file: path to the csv containing metadata on images (str)
    """
    root = os.getcwd()
    outputs_dir = os.path.join(root, 'model_outputs')
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
    gan_results = run_model(site, source, model_name, epoch, outputs_dir, num_images)
    print('GAN finished')
    print('Extracting Shorelines')
    shoreline_extraction_utils.process(gan_results,
                                       site,
                                       coords_file,
                                       outputs_dir,
                                       source)
    print('Shorelines Extracted')


def full_run(sitename,
             image_folder,
             new_image_folder,
             dataroot,
             model_name,
             old_csv,
             string,
             csv):
    split_and_resize(image_folder,
                     new_image_folder,
                     ext)

    run_and_process(sitename,
                    dataroot,
                    model_name,
                    csv)
