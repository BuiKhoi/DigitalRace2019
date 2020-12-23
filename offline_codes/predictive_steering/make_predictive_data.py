import os
import cv2
from shutil import copyfile

def get_files(img_folder):
    files = []
    for f1 in os.listdir(img_folder):
        f1_path = img_folder + f1 + '/'
        if not os.path.isdir(f1_path):
            continue
        for f2 in os.listdir(f1_path):
            f2_path = f1_path + f2+ '/'
            f2_files = []
            for file in os.listdir(f2_path):
                f2_files.append(f2_path + file)
            files.append(f2_files)
    return files

def get_time_millis(file_name):
    return file_name.split('/')[-1].split('x')[0]


if __name__ == '__main__':
    # image_folder = './images_data_left/'
    # target_folder = './images_data_predictive/'
    image_folder = './images_data_left/'
    target_folder = './images_data_predictive/'
    predictive_distance = 15 #frames

    images_files = get_files(image_folder)
    for idx, img_batch in enumerate(images_files):
        print('Processing idx {} of {}'.format(idx + 1, len(images_files)), end='\r')
        sorted_batch = img_batch.copy()
        sorted_batch.sort(key= lambda x: get_time_millis(x))
        for i in range(0, len(sorted_batch) - predictive_distance):
            current_file = sorted_batch[i]
            predictive_file = sorted_batch[i + predictive_distance]
            extension = current_file.split('.')[-1]
            mil, _, ste = current_file.split('/')[-1].split('.')[0].split('x')
            pred_ste = predictive_file.split('/')[-1].split('.')[0].split('x')[-1]
            
            new_file = mil + 'x' + pred_ste + 'x' + ste + '.' + extension
            copyfile(current_file, target_folder + new_file)