import math
import datetime
import os
from pathlib import Path
from scipy.spatial.distance import cdist
from torch.utils.data import Dataset
import torch.nn.functional as F
import os
import torch
import numpy as np
import cv2
from torch.utils.data import Dataset
from utils.preprocess_utils import get_perspective_mat, scale_homography, resize_aspect_ratio


class OxfordTrain(Dataset):
    """Sparse correspondences dataset."""
    def __init__(self, dataset_params, typ="train"):
        self.config = dataset_params
        self.train_path = os.path.join(dataset_params['dataset_path'], "train\\")
        self.files = []
        self.files += [self.train_path + f for f in os.listdir(self.train_path)]
        self.aug_params = dataset_params['augmentation_params']
        self.aspect_resize = dataset_params['resize_aspect']
        self.apply_aug = dataset_params['apply_color_aug']
        if self.apply_aug:
            import albumentations as alb
            self.aug_list = [alb.OneOf([alb.RandomBrightness(limit=0.4, p=0.6), alb.RandomContrast(limit=0.3, p=0.7)], p=0.6),
                             alb.OneOf([alb.MotionBlur(p=0.5), alb.GaussNoise(p=0.6)], p=0.5),
                             #alb.JpegCompression(quality_lower=65, quality_upper=100,p=0.4)
                             ]
            self.aug_func = alb.Compose(self.aug_list, p=0.65)

    def __len__(self):
        return len(self.files)

    def apply_augmentations(self, image1, image2):
        image1_dict = {'image': image1}
        image2_dict = {'image': image2}
        result1, result2 = self.aug_func(**image1_dict), self.aug_func(**image2_dict)
        return result1['image'], result2['image']

    def __getitem__(self, idx):
        resize = True
        file_name = self.files[idx]
        image = cv2.imread(file_name, cv2.IMREAD_GRAYSCALE)
        if self.aspect_resize:
            image = resize_aspect_ratio(image, self.config['image_height'], self.config['image_width'])
            resize = False
        height, width = image.shape[0:2]
        homo_matrix = get_perspective_mat(self.aug_params['patch_ratio'], width // 2, height // 2,
                                          self.aug_params['perspective_x'], self.aug_params['perspective_y'],
                                          self.aug_params['shear_ratio'], self.aug_params['shear_angle'],
                                          self.aug_params['rotation_angle'], self.aug_params['scale'],
                                          self.aug_params['translation'])
        warped_image = cv2.warpPerspective(image.copy(), homo_matrix, (width, height))
        if resize:
            orig_resized = cv2.resize(image, (self.config['image_width'], self.config['image_height']))
            warped_resized = cv2.resize(warped_image, (self.config['image_width'], self.config['image_height']))
        else:
            orig_resized = image
            warped_resized = warped_image
        if self.apply_aug:
            orig_resized, warped_resized = self.apply_augmentations(orig_resized, warped_resized)
        homo_matrix = scale_homography(homo_matrix, height, width, self.config['image_height'],
                                       self.config['image_width']).astype(np.float32)
        orig_resized = np.expand_dims(orig_resized, 0).astype(np.float32) / 255.0
        warped_resized = np.expand_dims(warped_resized, 0).astype(np.float32) / 255.0
        return orig_resized, warped_resized, homo_matrix


class Oxfordval(Dataset):
    def __init__(self, dataset_params):
        super(Oxfordval, self).__init__()
        self.config = dataset_params
        self.dataset_path = dataset_params['val_path']
        self.images_path = os.path.join(self.dataset_path, "val\\")
        self.txt_path = str(Path(__file__).parent / 'txt/R1M_val_images.txt')
        with open(self.txt_path, 'r') as f:
            self.image_info = f.readlines()

    def __len__(self):
        return len(self.image_info)

    def __getitem__(self, index: int):
        split_info = self.image_info[index].strip().split(' ')
        image_name = split_info[0]
        homo_info = list(map(lambda x: float(x), split_info[1:]))
        homo_matrix = np.array(homo_info).reshape((3,3)).astype(np.float32)
        image = cv2.imread(os.path.join(self.images_path, image_name), cv2.IMREAD_GRAYSCALE)
        height, width = image.shape[0:2]
        warped_image = cv2.warpPerspective(image.copy(), homo_matrix, (width, height))
        orig_resized = cv2.resize(image, (self.config['image_width'], self.config['image_height']))
        warped_resized = cv2.resize(warped_image, (self.config['image_width'], self.config['image_height']))
        homo_matrix = scale_homography(homo_matrix, height, width, self.config['image_height'], self.config['image_width']).astype(np.float32)
        orig_resized = np.expand_dims(orig_resized, 0).astype(np.float32) / 255.0
        warped_resized = np.expand_dims(warped_resized, 0).astype(np.float32) / 255.0
        return orig_resized, warped_resized, homo_matrix


def collate_batch(batch):
    list_elem = list(zip(*batch))
    orig_resized = torch.stack([torch.from_numpy(i) for i in list_elem[0]], 0)
    warped_resized = torch.stack([torch.from_numpy(i) for i in list_elem[1]], 0)
    homographies = torch.stack([torch.from_numpy(i) for i in list_elem[2]], 0)
    orig_warped_resized = torch.cat([orig_resized, warped_resized], 0)
    return [orig_warped_resized, homographies]