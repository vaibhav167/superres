import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch
import random
import glob
import subprocess
import os
from PIL import Image
import numpy as np
from keras.models import Sequential
from keras import layers
from keras import backend as K
from keras.callbacks import Callback
import tensorflow as tf
import wandb
from wandb.keras import WandbCallback
from model import Generator
from pdb import set_trace as bp

run = wandb.init(project='superres')
config = run.config

config.num_epochs = 10
config.batch_size = 32
config.input_height = 32
config.input_width = 32
config.output_height = 256
config.output_width = 256

val_dir = 'data/test'
train_dir = 'data/train'

# automatically get the data if it doesn't exist
if not os.path.exists("data"):
    print("Downloading flower dataset...")
    subprocess.check_output(
        "mkdir data && curl https://storage.googleapis.com/wandb/flower-enhance.tar.gz | tar xz -C data", shell=True)

config.steps_per_epoch = len(
    glob.glob(train_dir + "/*-in.jpg")) // config.batch_size
config.val_steps_per_epoch = len(
    glob.glob(val_dir + "/*-in.jpg")) // config.batch_size

def perceptual_distance(y_true, y_pred):
    """Calculate perceptual distance, DO NOT ALTER"""
    y_true = y_true * 255
    y_pred = y_pred * 255
    rmean = (y_true[:, :, :, 0] + y_pred[:, :, :, 0]) / 2
    r = y_true[:, :, :, 0] - y_pred[:, :, :, 0]
    g = y_true[:, :, :, 1] - y_pred[:, :, :, 1]
    b = y_true[:, :, :, 2] - y_pred[:, :, :, 2]

    return K.mean(K.sqrt((((512+rmean)*r*r)/256) + 4*g*g + (((767-rmean)*b*b)/256)))

class SRDataset(Dataset):
    def __init__(self, img_dir):
        self.input_filenames = glob.glob(img_dir + "/*-in.jpg")
    
    def __len__(self):
        return len(self.input_filenames)
    
    def __getitem__(self, idx):
        img_name = self.input_filenames[idx]
        small_img = np.array(Image.open(img_name)).transpose(2, 0, 1).astype(np.float32)/255.0
        large_img = np.array(Image.open(img_name.replace("-in.jpg", "-out.jpg"))).transpose(2, 0, 1).astype(np.float32)/255.0
        return small_img, large_img

big_keras, big_hat_keras = None, None
    
train_loader = DataLoader(SRDataset(train_dir), batch_size=config.batch_size, shuffle=True)
val_loader = DataLoader(SRDataset(val_dir), batch_size=config.batch_size, shuffle=True)

model_g = Generator()
model_g.to(torch.device("cuda"))
criterion_g = nn.MSELoss()
optimizer_g = torch.optim.Adam(model_g.parameters(), lr=1e-3)

wandb.watch(model_g)
sess = tf.Session()
for epoch in range(config.num_epochs):
    train_loss_g = 0
    num_batches = 0
    for small, big in train_loader:
        small, big = small.to(torch.device("cuda")), big.to(torch.device("cuda"))
        model_g.zero_grad()
        big_hat = model_g(small)
        loss_g = criterion_g(big_hat, big)
        train_loss_g += loss_g.item()
        loss_g.backward()
        optimizer_g.step()
        num_batches += 1
    
    train_loss_g /= num_batches
    
    # check validation loss after every epoch
    val_loss_g = 0
    percept_loss = 0
    num_batches = 0
    bp()
    for small, big in val_loader:
        small, big = small.to(torch.device("cuda")), big.to(torch.device("cuda"))
        with torch.no_grad():
            big_hat = model_g(small)
            loss_g = criterion_g(big_hat, big)
            val_loss_g += loss_g.item()
            big_keras = K.variable(value=big.to(torch.device("cpu")).numpy())
            big_hat_keras = K.variable(value=big_hat.to(torch.device("cpu")).numpy())
            percept_loss += K.eval(perceptual_distance(big_keras, big_hat_keras))
            num_batches += 1
    bp()        
    val_loss_g /= num_batches
    percept_loss /= num_batches
    # print train, val loss after every epoch
    print("Epoch {}/{}: train_loss_g = {}, val_loss_g = {}".format(epoch, 
            config.num_epochs, train_loss_g, val_loss_g))

    wandb.log({"Epoch": epoch, "Val loss": val_loss_g, "val_perceptual_loss": float(percept_loss)})    
    
    torch.save(model_g.state_dict(), "model_dict.pt")
    torch.save(model_g, "model.pt")