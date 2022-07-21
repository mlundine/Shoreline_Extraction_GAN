import numpy as np
import cv2
import os
import glob
from shutil import copyfile


def split_and_resize(image_folder, new_image_folder, ext='.jpg'):
    """
    splits input images into two square images
    then resizes so they both have width/height of 256
    inputs:
    image_folder: path to input jpegs (str)
    output_folder: path to output jpegs (str)
    """
    for image in glob.glob(image_folder + '\*' + ext):
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
        cmd = 'conda deactivate & conda activate labelme & labelme_json_to_dataset ' + file + ' -o ' + os.path.join(extra_outputs, os.path.splitext(os.path.basename(file))[0])
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
    for k in range(len(labels)):
        lab = np.random.choice(labels)
        name = os.path.splitext(os.path.basename(lab))[0]
        new_name = name+'.jpeg'
        new_lab = name+'.png'
        srcjpeg = os.path.join(annotation_folder, new_name)
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
        labels.remove(lab)
        copyfile(srcjpeg, dstjpeg)
        copyfile(srcpng, dstpng)
        i=i+1
        frac = i/total

def png_to_jpeg_edit(folder, outFolder):
    for im in glob.glob(folder + '\*.png'):
        img_gray = cv2.imread(im)[:,:,2]
        img_gray[img_gray>10] = 255
        img_gray[img_gray<255] = 0
        cv2.imwrite(os.path.join(outFolder, os.path.splitext(os.path.basename(im))[0]+'.jpeg'), img_gray)

def png_to_jpeg(in_folder, out_folder):
    ims = glob.glob(in_folder + '\*.png')
    for im in ims:
        name = os.path.splitext(os.path.basename(im))[0]
        new_name = name+'.jpeg'
        new_path = os.path.join(out_folder, new_name)
        copyfile(im, new_path)

def get_rgb_ims(label_folder, rgb_folder, save_rgb_folder):
    labs = glob.glob(label_folder+'\*fake.png')
    for lab in labs:
        labname = os.path.splitext(os.path.basename(lab))[0]
        idx = labname.find('_fake')
        labname = labname[0:idx]
        rgb_name = labname
        rgb_src = os.path.join(rgb_folder, rgb_name+'.jpeg')
        rgb_dst = os.path.join(save_rgb_folder, rgb_name+'.jpeg')
        try:
            copyfile(rgb_src, rgb_dst)
        except:
            continue
          

def png_fake_edit(folder, outFolder):
    labs = glob.glob(folder+'\*fake.png')
    for lab in labs:
        base = os.path.basename(lab)
        idx = base.find('_fake')

        newname = base[0:idx]+'.jpeg'
        img = cv2.imread(lab)
        img[img>250] = 255
        img[img<255] = 0
        cv2.imwrite(os.path.join(outFolder, newname), img)

def get_labs(rgb_folder, lab_folder, newlab_folder):
    for rgb in glob.glob(rgb_folder + '\*.jpeg'):
        name = os.path.basename(rgb)
        lab = os.path.join(lab_folder, name)
        newlab = os.path.join(newlab_folder, name)
        copyfile(lab, newlab)
