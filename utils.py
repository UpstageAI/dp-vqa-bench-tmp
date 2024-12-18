import os
from glob import glob

# write a code that reads all the image files in a folder
def get_image_path_list(path):
    image_paths = glob(path + '/*')

    return image_paths

def save_txt(image_path, image_dir, save_dir, content):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    filename = os.path.basename(image_path)
    basename = '.'.join(filename.split('.')[:-1])

    save_path = os.path.join(save_dir, basename + ".txt")

    with open(save_path, 'w') as f:
        f.write(content)
