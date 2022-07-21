import numpy as np
import cv2
import os
import glob


def augment(rgb_folder, lab_folder):
    images = glob.glob(rgb_folder + '\*.jpeg')
    for im in images:
        img = cv2.imread(im)
        name = os.path.basename(im)
        labpath = os.path.join(lab_folder, name)
        img_l = cv2.imread(labpath)

        name_no_ext = os.path.splitext(name)[0]
        
        #flip vertical
        vert = cv2.flip(img, 0)
        vert_l = cv2.flip(img_l, 0)
        save_im = os.path.join(rgb_folder, name_no_ext +'_vert.jpeg')
        save_im_l =  os.path.join(lab_folder, name_no_ext +'_vert.jpeg')
        cv2.imwrite(save_im, vert)
        cv2.imwrite(save_im_l, vert_l)
        #flip horizontal
        horizontal = cv2.flip(img,1)
        horizontal_l = cv2.flip(img_l,1)
        save_im = os.path.join(rgb_folder, name_no_ext +'_hor.jpeg')
        save_im_l = os.path.join(lab_folder, name_no_ext +'_hor.jpeg')
        cv2.imwrite(save_im, horizontal)
        cv2.imwrite(save_im_l, horizontal_l)
        #flip vert and horizont
        vert_horizontal = cv2.flip(img, -1)
        vert_horizontal_l = cv2.flip(img_l, -1)
        save_im = os.path.join(rgb_folder, name_no_ext +'_vert_hor.jpeg')
        save_im_l = os.path.join(lab_folder, name_no_ext +'_vert_hor.jpeg')
        cv2.imwrite(save_im, vert_horizontal)
        cv2.imwrite(save_im_l, vert_horizontal_l)
        #rotate 90 cw
        cw_rot = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        cw_rot_l = cv2.rotate(img_l, cv2.ROTATE_90_CLOCKWISE)
        save_im = os.path.join(rgb_folder, name_no_ext +'_cw.jpeg')
        save_im_l = os.path.join(lab_folder, name_no_ext +'_cw.jpeg')
        cv2.imwrite(save_im, cw_rot)
        cv2.imwrite(save_im_l, cw_rot_l)
        #rotate 90 ccw
        ccw_rot = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
        ccw_rot_l = cv2.rotate(img_l, cv2.ROTATE_90_COUNTERCLOCKWISE)
        save_im = os.path.join(rgb_folder, name_no_ext +'_ccw.jpeg')
        save_im_l = os.path.join(lab_folder, name_no_ext +'_ccw.jpeg')
        cv2.imwrite(save_im, ccw_rot)
        cv2.imwrite(save_im_l, ccw_rot_l)

augment(r'D:\stuff\A',
        r'D:\stuff\B')
