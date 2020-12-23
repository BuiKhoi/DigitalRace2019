import os
import cv2
import numpy as np
from shutil import copyfile, rmtree

VAL_RATIO = 0.3

data_folder = './segmented_data/'
target_folder = './train_data/'

if os.path.exists(target_folder + 'train'):
    rmtree(target_folder + 'train')
    
if os.path.exists(target_folder + 'val'):
    rmtree(target_folder + 'val')
    
os.mkdir(target_folder + 'train')
os.mkdir(target_folder + 'val')

images = [data_folder + f for f in os.listdir(data_folder)]

for image in images:
    rand = np.random.random()
    if rand < VAL_RATIO:
        file_path = target_folder + 'val/' + image.split('/')[-1]
    else:
        file_path = target_folder + 'train/' + image.split('/')[-1]
    copyfile(image, file_path)