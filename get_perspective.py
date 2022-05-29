import os
import numpy as np
import cv2
from utils.preprocess_utils import get_perspective_mat
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--image_dir', type=str)
opt = parser.parse_args()

np.random.seed(100)  # changed the seed if needed
image_dir = opt.image_dir
txt_file = open("txt/R1M_test_images.txt", 'w')  # path where the generated homographies should be stored

content = os.listdir(image_dir)
ma_fn = lambda x: float(x)
for kk, i in enumerate(content):
    if not os.path.splitext(i)[-1] in [".jpg", ".png"]:
        continue
    image = cv2.imread(os.path.join(image_dir, i))
    print(i)
    height, width = image.shape[0:2]
    homo_matrix = get_perspective_mat(0.85,center_x=width//2, center_y=height//2, pers_x=0.0008, pers_y=0.0008, shear_ratio=0.04, shear_angle=10, rotation_angle=25, scale=0.6, trans=0.6)
    res_image = cv2.warpPerspective(image.copy(), homo_matrix, (width, height))
    txt_file.write("{} {} {} {} {} {} {} {} {} {}\n".format(i, *list(map(ma_fn, list(homo_matrix.reshape(-1))))))
print('done')
