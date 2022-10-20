import os
import glob
import numpy as np
from shutil import copyfile

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
    root = os.path.dirname(os.getcwd())
    combine = os.path.join(root, 'pix2pix_modules', 'datasets','combine_A_and_B.py')
    cmd0 = 'conda deactivate & conda activate pix2pix_shoreline & '
    cmd1 = 'python ' + combine 
    cmd2 = ' --fold_A ' + foldA
    cmd3 = ' --fold_B ' + foldB
    cmd4 = ' --fold_AB ' + home
    cmd5 = ' --no_multiprocessing'
    full_cmd = cmd0+cmd1+cmd2+cmd3+cmd4+cmd5
    os.system(full_cmd)

setup_datasets(r'D:\New_Training_Images', r'shoreline_gan', r'D:\New_Training_Images\A',r'D:\New_Training_Images\B')


def split_datasets(pix2pix_training_folder, A_folder, B_folder):
    A = os.path.join(pix2pix_training_folder, 'A')
    B = os.path.join(pix2pix_training_folder, 'B')
    testA = os.path.join(pix2pix_training_folder, 'A', 'test')
    testB = os.path.join(pix2pix_training_folder, 'B', 'test')
    trainA = os.path.join(pix2pix_training_folder, 'A', 'train')
    trainB = os.path.join(pix2pix_training_folder, 'B', 'train')
    valA = os.path.join(pix2pix_training_folder, 'A', 'val')
    valB = os.path.join(pix2pix_training_folder, 'B', 'val')
    dirs = [A,B,testA,testB,trainA,trainB,valA,valB]
    for direc in dirs:
        try:
            os.mkdir(direc)
        except:
            pass


    labels = []
    for file in glob.glob(B_folder + '\*.jpeg'):
        labels.append(file)

    total = len(labels)
    i=0
    frac=0
    for k in range(len(labels)):
        lab = np.random.choice(labels)
        name = os.path.splitext(os.path.basename(lab))[0]
        new_name = name+'.jpeg'
        new_lab = name+'.jpeg'
        srcjpeg = os.path.join(A_folder, new_name)
        srcpng = lab
        if frac < 0.80:
            dstjpeg = os.path.join(trainA, new_name)
            dstpng = os.path.join(trainB, new_lab)
        elif frac < 0.90:
            dstjpeg = os.path.join(testA, new_name)
            dstpng = os.path.join(testB, new_lab)
        else:
            dstjpeg = os.path.join(valA, new_name)
            dstpng = os.path.join(valB, new_lab)
        labels.remove(lab)
        try:
            copyfile(srcjpeg, dstjpeg)
            copyfile(srcpng, dstpng)
        except:
            pass
        i=i+1
        frac = i/total

def train_model(model_name,
                model_type,
                dataroot,
                n_epochs = 10,
                n_epochs_decay = 5):
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
    cmd7 = ' --preprocess none'
    cmd8 = ' --checkpoints_dir ' + os.path.join(root, 'pix2pix_modules', 'checkpoints')
    cmd9 = ' --n_epochs ' + str(n_epochs)
    cmd10 = ' --n_epochs_decay ' + str(n_epochs_decay)
    cmd11 = ' --input_nc 3 --output_nc 1'
    cmd12 = ' --display_id 1'
    cmd13 = ' --lr 0.000002'
    cmd14 = ' --batch_size 1'
    
    full_cmd = cmd0+cmd1+cmd2+cmd3+cmd4+cmd5+cmd6+cmd7+cmd8+cmd9+cmd10+cmd11+cmd12+cmd13+cmd14
    
    os.system(full_cmd)
    print('Training Finished')

def continue_train_model(model_name,
                model_type,
                dataroot,
                epoch_count,
                n_epochs = 100,
                n_epochs_decay = 5):
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
    cmd7 = ' --preprocess none'
    cmd8 = ' --checkpoints_dir ' + os.path.join(root, 'pix2pix_modules', 'checkpoints')
    cmd9 = ' --n_epochs ' + str(n_epochs)
    cmd10 = ' --n_epochs_decay ' + str(n_epochs_decay)
    cmd11 = ' --input_nc 3 --output_nc 1'
    cmd12 = ' --display_id 1'
    cmd13 = ' --lr 0.000002'
    cmd14 = ' --batch_size 1'
    cmd15 = ' --epoch_count ' + str(epoch_count)
    cmd16 = ' --continue_train'
    
    full_cmd = cmd0+cmd1+cmd2+cmd3+cmd4+cmd5+cmd6+cmd7+cmd8+cmd9+cmd10+cmd11+cmd12+cmd13+cmd14+cmd15+cmd16
    
    os.system(full_cmd)
    print('Training Finished')
