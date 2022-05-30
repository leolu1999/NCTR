import os
import shutil
import numpy as np
import argparse


def makedir(new_dir):
    if not os.path.exists(new_dir):
        os.makedirs(new_dir)


parser = argparse.ArgumentParser()
parser.add_argument('--input_dir', type=str)
parser.add_argument('--output_dir', type=str)
opt = parser.parse_args()

makedir('txt')
val_test_index = np.random.randint(0, 1000000, size=1000)
np.savetxt("txt/val_index.txt", val_test_index[:500])
np.savetxt("txt/test_index.txt", val_test_index[500:])
val_index = np.loadtxt("txt/val_index.txt")
test_index = np.loadtxt("txt/test_index.txt")

input_dir = opt.input_dir
output_dir = opt.output_dir

train_dir = os.path.join(output_dir, "train")
val_dir = os.path.join(output_dir, "val")
test_dir = os.path.join(output_dir, "test")

j = 0
for root, dirs, files in os.walk(input_dir):
    for sub_dir in dirs:
        imgs = os.listdir(os.path.join(root, sub_dir))
        imgs = list(filter(lambda x: x.endswith('.jpg'), imgs))
        img_count = len(imgs)
        for i in range(img_count):
            if j in val_index:
                out_dir = os.path.join(val_dir)
                print('valid:{}'.format(j))
            elif j in test_index:
                out_dir = os.path.join(test_dir)
                print('test:{}'.format(j))
            else:
                out_dir = os.path.join(train_dir)
                print('train:{}'.format(j))
            makedir(out_dir)
            target_path = os.path.join(out_dir, imgs[i])
            src_path = os.path.join(input_dir, sub_dir, imgs[i])
            shutil.copy(src_path, target_path)
            j = j + 1
