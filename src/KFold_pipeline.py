# -*- coding: utf-8 -*-

# This script runs a KFold pipeline. This means K trainings for each model. At
# the end of the K trainings it records the mean dice coefficent of all of
# them. You can control K with the variable KFOLD at the section 'Global
# variables and parameters'

###############################################################################
# Importation of libraries
###############################################################################

import os
from imutils import paths
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms.functional import to_tensor
import cv2
from PIL import Image
import numpy as np
import random
from sklearn.model_selection import KFold
import pathlib
import sys
import albumentations as A

SEED = 0
np.random.seed(SEED)
torch.manual_seed(SEED)
random.seed(SEED)

PROJECT_DIR = str(pathlib.Path().resolve())
sys.path.append(PROJECT_DIR)
import custom_utils as h


###############################################################################
# Global variables and parameters
###############################################################################

# Paths to the datasets
ds = 'data'
IMAGE_PATH = os.path.join(PROJECT_DIR, ds + '/img')
MASK_PATH = os.path.join(PROJECT_DIR, ds + '/msk')

# Parameters
NUM_CLASS = 3
batch_size = 6
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
IM_SIZE = 224
KFOLD = 10
num_workers = 0


###############################################################################
# The dataset class
###############################################################################

class SegDataset(Dataset):
    def __init__(self, image_paths, mask_paths, transformations=None,
                 num_class=2):
        # Store the image and mask filepaths, and augmentation transforms
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.transforms = transformations
        self.num_class = num_class

    def __len__(self):
        # Return the number of total samples contained in the dataset
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Grab the image path from the current index
        im_path = self.image_paths[idx]
        # Load the image with PIL as numpy array
        im = np.array(Image.open(im_path))
        # Read mask with PIL as numpy array
        ms_path = self.mask_paths[idx]
        mk = np.array(Image.open((ms_path)))

        # Apply albumentations transforms
        augmented = self.transforms(image=im, mask=mk)
        image = augmented['image']
        mask = augmented['mask']

        # Convert flat mask to deep mask (one channel per class)
        mask = self.__deep_mask__(mask)

        # return a tuple of the image and its mask as tensors
        return (to_tensor(image), to_tensor(mask))

    def __deep_mask__(self, mask):
        # Convert a flat grayscale mask into a channeled grayscale mask

        # Convert a np mask with a grayscale format (H x W) with value range
        # 0-255 (equally spaced; i.e. 2 classes + background would be 0, 127,
        # 255) to a numpy mask (H x W x C) with C equal to number of channels
        # (classes) with 255 in the active pixels and 0 otherwise (uint8
        # format)

        # Number of different levels
        nlevels = self.num_class

        # Mask arrives with 255 gray levels equaly spaced
        mask = np.round(mask / (255 / nlevels))

        # Leave each level in a channel
        masks = []
        for k in range(nlevels):
            base = np.zeros_like(mask, dtype=np.uint8)
            ix = np.argwhere(mask == (k + 1))
            base[ix[:, 0], ix[:, 1]] = 255
            masks.append(base)

        # Stack all channels
        return np.stack(masks, axis=2)


###############################################################################
# Image transformations for augmentation
###############################################################################

# Define transforms for train and validation. If you don't want to apply any
# transformation to the data, apply transform_test to both datasets when
# defining them in section "Loop through models and nested loop through K-fold"
transform_train = A.Compose([A.Rotate(limit=45,
                                      border_mode=cv2.BORDER_CONSTANT,
                                      value=0,
                                      mask_value=0),
                             A.RandomCrop(width=IM_SIZE, height=IM_SIZE),
                             A.VerticalFlip(p=0.5),
                             A.HorizontalFlip(p=0.5)])

transform_test = A.Compose([A.CenterCrop(width=IM_SIZE, height=IM_SIZE)])


###############################################################################
# Preparing datasets
###############################################################################

# Load the image and mask filepaths in a sorted manner
image_paths = sorted(list(paths.list_images(IMAGE_PATH)))
mask_paths = sorted(list(paths.list_images(MASK_PATH)))

# Check for completeness
for k in range(len(image_paths)):
    imname = os.path.basename(image_paths[k]).split('.')[0]
    mkname = os.path.basename(mask_paths[k]).split('.')[0]
    if imname != mkname:
        raise TypeError(f"Names inconsistent. Image: ({imname}) Mask: {mkname}")

# Output as numpy arrays
image_paths, mask_paths = np.array(image_paths), np.array(mask_paths)

###############################################################################
# Loop through models and nested loop through K-fold
###############################################################################

# Parameter for indexes array
idxs = np.arange(image_paths.size)

# Loop through all models
for m_option in range(1, 10):

    # K-fold validation loop
    splits = KFold(n_splits=KFOLD, shuffle=True, random_state=42)
    dice_list = []
    model, arch, encoder_name, optimizer, scheduler = h.load_model(
        m_option, NUM_CLASS)
    # Tweak hiperparameters of optimizer and scheduler now if desired
    print(f"Working on {arch} {encoder_name}")

    # The main loop
    for fold, (train_idx, val_idx) in enumerate(splits.split(idxs)):

        print(f"Fold {fold + 1}")
        if fold >= 9:
            print('\n\n###########')
            print('# Fold {} #'.format(fold + 1))
            print('###########')
        else:
            print('\n\n##########')
            print('# Fold {} #'.format(fold + 1))
            print('##########')

        # Temp datasets for this fold
        train_set = SegDataset(image_paths[train_idx].tolist(),
                               mask_paths[train_idx].tolist(),
                               transform_train,  # transform_test,
                               NUM_CLASS)
        test_set = SegDataset(image_paths[val_idx].tolist(),
                              mask_paths[val_idx].tolist(),
                              transform_test,
                              NUM_CLASS)

        # Temp dataloaders for this fold
        train_loader = DataLoader(train_set,
                                  batch_size=batch_size,
                                  num_workers=num_workers)
        test_loader = DataLoader(test_set,
                                 batch_size=batch_size,
                                 num_workers=num_workers)
        Kloaders = {'train': train_loader, 'valid': test_loader}

        # Do the training
        n_epochs = 30
        model, training_log = h.train_model(model=model,
                                            optimizer=optimizer,
                                            scheduler=scheduler,
                                            loaders=Kloaders,
                                            num_epochs=n_epochs,
                                            checkpoint={},
                                            verbose=1)

        # Training info
        val_loss_list = [m['valid_loss'] for m in training_log]
        min_loss_ix = np.argmin(val_loss_list)
        min_loss = val_loss_list[min_loss_ix]
        dice_val_list = [m['valid_dice'] for m in training_log]
        dice_val = dice_val_list[min_loss_ix]
        dice_list.append(dice_val)
        print(f"\n\nFOLD {fold + 1} TRAINING SUMMARY:")
        print("-------------------------")
        print(f"Dice coeffcicient of best model in fold {fold + 1}: {dice_val}")
        print(f"Min validation loss in fold {fold + 1}: {min_loss}")

    # Final info of the training process
    s = f"Dice coefficient: {np.mean(dice_list)}\n\n"
    print(s)
    with open('output/comp.txt', 'a') as file:
        file.write(f"Model: {arch} {encoder_name}\n")
        file.write(s)
