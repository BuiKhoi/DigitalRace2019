import os

def get_all_files(image_folder, generated_images):
    files = []
    for f in os.listdir(image_folder):
        if os.path.isdir(image_folder + f):
            files.extend(get_all_files(image_folder + f + '/', generated_images))
        else:
            if any(b in f for b in ['png', 'jpg']):
                file_path = image_folder + f
                if file_path not in generated_images:
                    files.append(file_path)
    return files

def get_files(image_folder, generated_images_file):
    print(generated_images_file)
    with open(generated_images_file, 'r') as images_files:
        generated_images = images_files.read().split('\n')[:-1]
    files = get_all_files(image_folder, generated_images)
    print('Loaded {} files'.format(len(files)))
    with open(generated_images_file, 'a') as images_files:
        for f in files:
            images_files.write(f)
            images_files.write('\n')
    return files

if __name__ == '__main__':
    image_folder = './image_data/'
    generated_files = './generated_images.txt'
    files = get_files(image_folder, generated_files)