# -*- coding: utf-8 -*-

# This script runs a training of a segmentation model with train, validation
# and test sets.
# Afterwards it makes an example of extraction and comparison of the
# graphical features.

# Pay attention to the section "Global variables and parameters", where you
# will be able to customize the execution.

# Moreover, there are some blocks that can be enabled or disabled by changing
# a constant. These blocks start with: if True (or if False)

###############################################################################
# Importation of libraries and set up
###############################################################################

import os
from imutils import paths
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms.functional import to_tensor
from PIL import Image
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pathlib
import sys
import albumentations as A

# Data analysis of features with pyradiomics
from radiomics import featureextractor
import pandas as pd
from sklearn import manifold
from sklearn.metrics import euclidean_distances

SEED = 0
np.random.seed(SEED)
torch.manual_seed(SEED)

PROJECT_DIR = str(pathlib.Path().resolve())
sys.path.append(PROJECT_DIR)
import custom_utils as h


###############################################################################
# Global variables and parameters
###############################################################################

# Paths to the datasets
ds = 'data'  # relative path from PROJECT_DIR
IMAGE_PATH = os.path.join(PROJECT_DIR, ds + '/img')
MASK_PATH = os.path.join(PROJECT_DIR, ds + '/msk')

# Parameters
NUM_CLASS = 3
batch_size = 4
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
IM_SIZE = 224
VALID_SIZE = 0.2
TEST_SIZE = 0.2


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

# Define transforms for train/valid/test
transform_train = A.Compose([A.Rotate(limit=45,
                                      border_mode=cv2.BORDER_CONSTANT,
                                      value=0,
                                      mask_value=0),
                             A.RandomCrop(width=IM_SIZE, height=IM_SIZE),
                             A.VerticalFlip(p=0.5),
                             A.HorizontalFlip(p=0.5)])

transform_test = A.Compose([A.CenterCrop(width=IM_SIZE, height=IM_SIZE)])


###############################################################################
# Loading datasets
###############################################################################

# load the image and mask filepaths in a sorted manner
image_paths = sorted(list(paths.list_images(IMAGE_PATH)))
mask_paths = sorted(list(paths.list_images(MASK_PATH)))

# Check for completeness
for k in range(len(image_paths)):
    imname = os.path.basename(image_paths[k]).split('.')[0]
    mkname = os.path.basename(mask_paths[k]).split('.')[0]
    if imname != mkname:
        raise TypeError(f"Names inconsistent. Image: ({imname}) Mask: {mkname}")

# Train, validation and test split
n_idx = len(image_paths)
idx_list = list(range(0, n_idx))
np.random.shuffle(idx_list)
n1 = int((1 - VALID_SIZE - TEST_SIZE) * n_idx)
n2 = n1 + int(VALID_SIZE * n_idx)

# Train paths
image_train_paths = np.array(image_paths)[idx_list[:n1]].tolist()
mask_train_paths = np.array(mask_paths)[idx_list[:n1]].tolist()

# Validation paths
image_valid_paths = np.array(image_paths)[idx_list[n1:n2]].tolist()
mask_valid_paths = np.array(mask_paths)[idx_list[n1:n2]].tolist()

# Test paths
image_test_paths = np.array(image_paths)[idx_list[n2:]].tolist()
mask_test_paths = np.array(mask_paths)[idx_list[n2:]].tolist()

# Define the three basic datasets
train_set = SegDataset(image_train_paths,
                       mask_train_paths,
                       transform_train,
                       NUM_CLASS)

valid_set = SegDataset(image_valid_paths,
                       mask_valid_paths,
                       transform_test,
                       NUM_CLASS)

test_set = SegDataset(image_test_paths,
                      mask_test_paths,
                      transform_test,
                      NUM_CLASS)

datasets = {'train': train_set, 'valid': valid_set, 'test': test_set}


###############################################################################
# Dataloaders
############################################################################

num_workers = 0
shuffle = False
tbs = 8  # Test batch size
dataloaders = {'train': DataLoader(train_set,
                                   batch_size=batch_size,
                                   shuffle=shuffle,
                                   num_workers=num_workers),
               'valid': DataLoader(valid_set,
                                   batch_size=batch_size,
                                   shuffle=shuffle,
                                   num_workers=num_workers),
               'test': DataLoader(test_set,
                                  batch_size=tbs,
                                  shuffle=False,
                                  num_workers=num_workers)}


###############################################################################
# Check first image and mask of one batch
###############################################################################

if True:

    # Get a batch of training data
    inputs, masks = next(iter(dataloaders['train']))

    # Convert tensor image into numpy and display it
    im = h.to_npimage(inputs[0])
    plt.imshow(im)

    # Convert tensor mask into numpy and display it
    msk = h.flat_mask(masks[0])
    plt.imshow(msk, cmap='gray')


###############################################################################
# Load the segmentation model
###############################################################################

# MODEL OPTIONS
# 1 - DeepLabV3Plus with encoder from efficientnet b6
# 2 - MAnet with encoder from timm-efficientnet b6
# 3 - UNet++ with encoder from efficientnet b6
# 4 - UNet with encoder from timm-efficientnet b6
# 5 - U¹Net
# 6 - PAN with encoder from timm-efficientnet b6
# 7 - FPN with encoder from timm-efficientnet b6
# 8 - PSPNet with encoder from timm-efficientnet b6
# 9 - Linknet with encoder from timm-efficientnet b6

m_option = 3

# Load the model
model, arch, encoder_name, optimizer, scheduler = h.load_model(m_option,
                                                               NUM_CLASS)
# Tweak hiperparameters of optimizer and scheduler now if desired

# Check if a saved model is requiered
LOAD_TRAINED = False
FILENAME = 'model.pth' if not LOAD_TRAINED else 'unetpp_2ldice.pth'

if LOAD_TRAINED:
    # Customize it as desired!!!!

    # Load the parameters that you want. Take into account that by default
    # the statedict of the optimizer and scheduler are pre-loaded, and
    # this could be different than what you expect.

    # Load checkpoint (the dict saved by 'save_tr_model')
    checkpoint = torch.load(FILENAME, map_location=torch.device('cpu'))

    # Pass the current model to cpu
    model = model.to('cpu')

    # Load the state_dict of the model
    model.load_state_dict(checkpoint['model_state_dict'])

    # Pass the model to DEVICE
    model = model.to(DEVICE)

    # Load state_dict of the optimizador and scheduler
    optimizer.load_state_dict(checkpoint['opt_state_dict'])
    scheduler.load_state_dict(checkpoint['sche_state_dict'])

    # Load the training log
    training_log = checkpoint['training_log']

    # Display data about the model loaded
    tsav = checkpoint['time']
    print('Saved on year: {}, month: {}, day:{}, time: {}:{}'.format(
        tsav.tm_year, tsav.tm_mon, tsav.tm_mday, tsav.tm_hour, tsav.tm_min))


###############################################################################
# Train the model
###############################################################################

# Do the training
if not LOAD_TRAINED:
    n_epochs = 35
    model, training_log = h.train_model(model=model,
                                        optimizer=optimizer,
                                        scheduler=scheduler,
                                        loaders=dataloaders,
                                        num_epochs=n_epochs,
                                        checkpoint={},
                                        verbose=1,
                                        lf=1)


###############################################################################
# See some results
###############################################################################

if True:

    # Plot training log
    h.plot_training_log(training_log)

    # Get a test batch
    model.to('cpu')
    model.eval()
    inputs, masks = next(iter(dataloaders['test']))
    # inputs = inputs.to(DEVICE)
    # masks = masks.to(DEVICE)

    # Predict
    pred = model(inputs)
    pred = torch.sigmoid(pred)

    # Leave everything at cpu
    inputs, masks, pred = inputs.cpu(), masks.cpu(), pred.cpu()

    # View some results
    for i in range(tbs):
        # Image
        im = h.to_npimage(inputs[i])

        # Original mask
        msk = h.flat_mask(masks[i])

        # Predicted mask
        prd = h.flat_mask(pred[i])

        # Show comparison
        h.show_compare(im, msk, prd)


###############################################################################
# Test the model
###############################################################################

if True:

    # Run the test
    test_metrics = h.test(dataloaders['test'], model)

    # Show dice coefficient on test dataset
    print('\n\nDice coefficient: {:.2f}%'.format(100 * test_metrics['test_dice']))


###############################################################################
# Analysis of features with pyradiomics
###############################################################################

###############################################################################
# Feature extractors
###############################################################################

# Select settings of the extractors
K_NUCLEUS, K_CYTO, K_ENV, K_BACK = 255, 170, 85, 0
settings_nuc, settings_cyt = {}, {}
settings_nuc['force2D'], settings_cyt['force2D'] = True, True
settings_nuc['label'], settings_cyt['label'] = K_NUCLEUS, K_CYTO

# Define extractors
extr_nuc = featureextractor.RadiomicsFeatureExtractor(**settings_nuc)
extr_cyt = featureextractor.RadiomicsFeatureExtractor(**settings_cyt)

# Configure feature classes
extr_nuc.disableAllFeatures()
extr_cyt.disableAllFeatures()
extr_nuc.enableFeatureClassByName('firstorder')
extr_nuc.enableFeatureClassByName('shape2D')
extr_cyt.enableFeatureClassByName('firstorder')
extr_cyt.enableFeatureClassByName('shape2D')


###############################################################################
# Labels of images depending on cell type
###############################################################################

# Load the data (type, image and mask paths, and graphic features of cells)
fname = 'df.csv'
df = pd.read_csv(os.path.join(PROJECT_DIR, "pyrad_data", fname), sep="\t")
colnames = ['index'] + list(df.columns)[1:]
df.columns = colnames
df.set_index('index')

# Define cell types in the dataset
cells = df.cell_type.unique()

# Get one single image as example
np.random.seed(SEED)
random_ix = np.random.choice(np.arange(len(image_paths)))
test_im_path = image_paths[random_ix]
test_mk_path = mask_paths[random_ix]

# Get cell type of test image (if available), but don't use it to classify
for cell in cells:
    if cell in test_im_path:
        cell_type = cell

# Extract dictionary of features of the test image
test_feat = h.extract_features(test_im_path, test_mk_path, extr_nuc, extr_cyt)


###############################################################################
# Load dataframes from feature extraction process
###############################################################################

# This block has to be run when 'feature_extractor.py' has created the .csv
# files. Otherwise, you won't be able to do the analysis.

if True:
    # Load dataframes with cells data and summaries
    name = "monocyte_df.csv"
    fp = os.path.join(PROJECT_DIR, "pyrad_data", name)
    monocyte_df = pd.read_csv(fp, sep="\t")
    monocyte_df.columns = colnames
    monocyte_df.set_index('index')

    name = "lymphocyte_df.csv"
    fp = os.path.join(PROJECT_DIR, "pyrad_data", name)
    lymphocyte_df = pd.read_csv(fp, sep="\t")
    lymphocyte_df.columns = colnames
    lymphocyte_df.set_index('index')

    name = "hcl_df.csv"
    fp = os.path.join(PROJECT_DIR, "pyrad_data", name)
    hcl_df = pd.read_csv(fp, sep="\t")
    hcl_df.columns = colnames
    hcl_df.set_index('index')

    name = "apl_df.csv"
    fp = os.path.join(PROJECT_DIR, "pyrad_data", name)
    apl_df = pd.read_csv(fp, sep="\t")
    apl_df.columns = colnames
    apl_df.set_index('index')

    name = "monocyte_df_summary.csv"
    fp = os.path.join(PROJECT_DIR, "pyrad_data", name)
    monocyte_s = pd.read_csv(fp, sep="\t").iloc[0, 1:]
    monocyte_dfs = monocyte_s.to_frame().T

    name = "lymphocyte_df_summary.csv"
    fp = os.path.join(PROJECT_DIR, "pyrad_data", name)
    lymphocyte_s = pd.read_csv(fp, sep="\t").iloc[0, 1:]
    lymphocyte_dfs = lymphocyte_s.to_frame().T

    name = "hcl_df_summary.csv"
    fp = os.path.join(PROJECT_DIR, "pyrad_data", name)
    hcl_s = pd.read_csv(fp, sep="\t").iloc[0, 1:]
    hcl_dfs = hcl_s.to_frame().T

    name = "apl_df_summary.csv"
    fp = os.path.join(PROJECT_DIR, "pyrad_data", name)
    apl_s = pd.read_csv(fp, sep="\t").iloc[0, 1:]
    apl_dfs = apl_s.to_frame().T


###############################################################################
# Multidimensional scaling (MDS)
###############################################################################

# If you have done this analysis once, you don't need to do it again. The
# result is saved in the folder 'output' and will be loaded. Otherwise, change
# the option in the next active line to: `if True:`

# Load or do the MDS analysis
if True:
    # Extract columns with shape data
    shape = list(np.arange(4, 13)) + list(np.arange(31, 40)) + list(
        np.arange(58, 66))
    data = df.iloc[:, shape]

    similarities = euclidean_distances(data)

    seed = np.random.RandomState(seed=SEED)
    mds = manifold.MDS(n_components=2,
                       max_iter=10000,
                       eps=1e-12,
                       random_state=seed,
                       n_init=10,
                       dissimilarity="precomputed",
                       n_jobs=1,
                       metric=False)

    pos = mds.fit_transform(similarities)
    np.save(os.path.join(PROJECT_DIR, 'output', 'MDS.npy'), pos)
else:
    # Load the numpy array of the analysis
    pos = np.load(os.path.join(PROJECT_DIR, 'output', 'MDS.npy'))

# Figure
fig = plt.figure(1)
ax = plt.axes([0., 0., 1., 1.])
s = 10

# Indexes of each group
idx_1 = list(monocyte_df["index"])
idx_2 = list(lymphocyte_df["index"])
idx_3 = list(hcl_df["index"])
idx_4 = list(apl_df["index"])

# Scatterplots
plt.scatter(pos[idx_4, 0], pos[idx_4, 1], color='navy', alpha=1, s=s, lw=0.1,
            label=cells[3])
plt.scatter(pos[idx_3, 0], pos[idx_3, 1], color='turquoise', alpha=1, s=s,
            lw=0.1, label=cells[2])
plt.scatter(pos[idx_2, 0], pos[idx_2, 1], color='darkorange', alpha=1, s=s,
            lw=0.1, label=cells[1])
plt.scatter(pos[idx_1, 0], pos[idx_1, 1], color='red', alpha=1, s=s, lw=0.1,
            label=cells[0])
plt.scatter(pos[random_ix, 0], pos[random_ix, 1], color='green', alpha=1,
            s=100, lw=1, label='NEW')

plt.legend(scatterpoints=1, loc=5, shadow=False)

# similarities = similarities.max() / similarities * 100
# similarities[np.isinf(similarities)] = 0


###############################################################################
# Graphical comparison of features with means of groups
###############################################################################

if True:
    # Get image and transform to grayscale
    # Load the image with PIL as numpy array
    im_PIL = Image.open(test_im_path).convert('L')
    plt.imshow(np.array(im_PIL), cmap='gray')

    # Extract columns with shape data. It can be done with any feature set.
    # This example is done with monocyte dataframe
    shape = list(np.arange(0, 9)) + list(np.arange(27, 36)) + list(
        np.arange(54, 62))
    group_data = dict(monocyte_dfs.iloc[:, shape])

    # Get new feature set and extract same features. At the end we have a
    # dataframe with two rows. First row is the current image feature, and
    # second row is the summary (means) features of the group considered.
    data = pd.DataFrame()
    test_feat_d = {**{'index': 0}, **test_feat}
    data = data.append(test_feat_d, ignore_index=True)
    data = data.iloc[:, 4:]
    data = data.iloc[:, shape]
    data = data.append(dict(group_data), ignore_index=True)

    # Labels
    labs = monocyte_dfs.columns[shape]

    # Comparison of plots
    color = 'blue'
    fig, axes = plt.subplots(1, figsize=(20, 20))
    plt.plot(data.T, label=["Current", "Monocytes means"])
    plt.ylabel('Y tal', color=color)
    axes.set_yscale('log')
    axes.set_xticks(ticks=labs)
    axes.legend(loc="upper left")
    axes.grid(True)
    axes.grid(color='b', alpha=0.5, linestyle='dashed', linewidth=1)
    axes.set_title('Current features and mean features from monocytes',
                   color=color);
    axes.tick_params(axis='x', colors=color, rotation=90)
    axes.tick_params(axis='y', colors=color)
    plt.show()
