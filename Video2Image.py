import os
import shutil

TRAIN_ROOT = 'data/train_folder'
VALIDATION_ROOT = 'data/validation_folder/'
videos_folder = 'data/videos_folder/'
total_frames = 592
split_ratio = 4
split_point = 500
recreate = False

if recreate:
    shutil.rmtree(TRAIN_ROOT)
    shutil.rmtree(VALIDATION_ROOT)
    os.makedirs('data/train_folder')
    os.makedirs('data/validation_folder')

def _img_path(pig_class, img_id, folder):
    if img_id == None:
        return folder + str(pig_class)
    else:
        return folder + str(pig_class) + '/' + ('image%d-%04d' %
                                                (pig_class, img_id)) + '.jpg'

for pig_class in range(1,31):
    for img_id in range(1, total_frames):
        if img_id >= split_point:
            try:
                shutil.copy(_img_path(pig_class, img_id, videos_folder), _img_path(pig_class, img_id, VALIDATION_ROOT))
            except:
                os.makedirs(_img_path(pig_class, None, VALIDATION_ROOT))
