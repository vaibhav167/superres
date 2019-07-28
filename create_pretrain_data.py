import os
import subprocess
from PIL import Image
import numpy as np
from pdb import set_trace as bp

N_TRAIN = 4500

if not os.path.exists("data"):
    print("Downloading pascal voc dataset...")
    subprocess.check_output(
        "mkdir data && wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar | tar xz -C data", shell=True)

image_names = os.listdir("data/VOCdevkit/VOC2007/JPEGImages")

# copy to train folder
os.system("mkdir data/train")
shapes = []
num_train = 0
for name in image_names[:N_TRAIN]:
    img = Image.open(os.path.join("data/VOCdevkit/VOC2007/JPEGImages", name))
    height, width = img.size
    if height < 256 or width < 256:
        continue
    img_hr = img.crop(box=(0, 0, 256, 256))
    img_lr = img_hr.resize((32, 32), Image.BICUBIC)

    img_hr.save(os.path.join("data/train", name.split('.')[0] + ('-out.jpg')))
    img_lr.save(os.path.join("data/train", name.split('.')[0] + ('-in.jpg')))
    num_train += 1

print("Number of training images saved = ", num_train)

# copy to test folder
os.system("mkdir data/test")
shapes = []
num_test = 0
for name in image_names[N_TRAIN:]:
    img = Image.open(os.path.join("data/VOCdevkit/VOC2007/JPEGImages", name))
    height, width = img.size
    if height < 256 or width < 256:
        continue
    img_hr = img.crop(box=(0, 0, 256, 256))
    img_lr = img_hr.resize((32, 32), Image.BICUBIC)

    img_hr.save(os.path.join("data/test", name.split('.')[0] + ('-out.jpg')))
    img_lr.save(os.path.join("data/test", name.split('.')[0] + ('-in.jpg')))
    num_test += 1
print("Number of testing images saved = ", num_test)