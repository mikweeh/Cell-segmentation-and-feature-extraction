#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os

import torch
import torch.optim as optim
from torch.optim import lr_scheduler
import time
import copy
import numpy as np
import matplotlib.pyplot as plt
import segmentation_models_pytorch as smp
from collections import defaultdict
import torch.nn.functional as F
from u2net.u2net import U2NET
from PIL import Image
import SimpleITK as sitk
import six

# DEVICE - Global variable. Use GPU if possible.
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

###############################################################################
# Function: list_files
###############################################################################

# This function extract a list of full path of all the files present in folder
# 'path' or in any of its subfolders.

# Inputs:
#     path: (string) Path of base folder

# Outputs:
#     path_list: (list) List of full paths of all the files


def list_files(path):
    # Extract list of full path of all files in 'path'
    paths_list = []
    for root, dirs, files in os.walk(path):
        if len(files) == 0:
            continue
        else:
            for file in files:
                full_path = os.path.join(root, file)
                paths_list.append(full_path)
    return sorted(paths_list)


###############################################################################
# Function: to_npimage
###############################################################################

# Convert a torch tensor image with shape (C x H x W) to numpy array image of
# shape (H x W x C). C stands for channel, H for height and W for width.

# Inputs:
#     tensor_im: (tensor) Torch tensor with 3 dimensions (C x H x W)

# Outputs:
#     - (np-array) Array of three dimensions (H x W x C)


def to_npimage(tensor_im):
    # From tensor with shape (C x H x W) to numpy with shape (H x W x C)
    return tensor_im.numpy().transpose((1, 2, 0))


###############################################################################
# Function: flat_mask
###############################################################################

# Convert a deep mask into a flat mask.
# Convert a torch tensor with dimensions (C x H x W) being C>1 (i.e. multiple
# channels) in a single channel grayscale image numpy array with data type
# uint8 and dimensions (C x W). Deep masks can't be shown in matplotlib, but
# flat masks can.

# Inputs:
#     tensor_mask: (tensor) Torch tensor with 3 dimensions (C x H x W)
#     level: (float) Number between 0 and 1. Tensor values lower than level
#             will be coerced to 0, and to 1 otherwise. Default: 0.5

# Outputs:
#     mask_gray: (np-array) Array of two dimensions (H x W )


def flat_mask(tensor_mask, level=0.5):
    # Convert a deep mask into a flat mask

    # Convert a torch tensor with a channel for each class into a numpy
    # array grayscale image with one single channel and data type uint8
    mask = tensor_mask.data.cpu().numpy().transpose((1, 2, 0))
    apply = np.vectorize(lambda x: 0 if x < level else 1)
    mask = apply(mask)

    # Number of different levels
    nlevels = mask.shape[2]

    # Calculate values as integer levels
    for k in range(nlevels):
        mask[:, :, k] = mask[:, :, k] * (k + 1)

    # Scale to 255 and format uint8
    mask_gray = np.max(mask, axis=2) * (255 / nlevels)
    mask_gray = np.array(mask_gray, dtype=np.uint8)

    return mask_gray


###############################################################################
# Function: save_tr_model
###############################################################################

# Save trained model
# Save all the basic training model data (state_dict of model, optimizer and
# scheduler). It alsa saves the training_log and the local time.

# Inputs:
#     model: (torch model) The trained model.
#     optimizer: (optimizer) The optimizer object.
#     scheduler: (scheduler) The scheduler.
#     training_log: (list) Training log as created in 'train_model' function.
#     filename: (string) Name of the file to save the data.

# Outputs:
#     None - The file is saved in the current folder

def save_tr_model(model, optimizer, scheduler, training_log,
                  filename='model.pth', *args, **kwargs):
    model.to('cpu')
    custom_dict = {'model_state_dict': model.state_dict(),
                   'opt_state_dict': optimizer.state_dict(),
                   'sche_state_dict': scheduler.state_dict(),
                   'training_log': training_log,
                   'time': time.localtime()}

    torch.save(custom_dict, filename)
    model.to(DEVICE)


###############################################################################
# Function: dice_loss_fcn
###############################################################################

# Dice loss function
# Return loss function as the inverse of dice coefficient. Dice coefficient
# goes from 0 to 1, being 1 the best possible result and 0 the worst. This
# function, as a loss function, returns a value that is better as the loss is
# closer to 0. The output is calculated as 1 - dice_coefficient


# Inputs:
#     pred: (torch tensor) Predicted mask of shape (C x H x W)
#     target: (torch tensor) Target mask of same shape
#     smooth: (float) Smooth coefficient to avoid division by 0
#     layers_from: (integer) First layer (channel) to take into account in
#           the calculation of the Dice coefficient. Default is 0.

# Outputs:
#     - (float) Number between 0 and 1. Calculated as (1 - dice coefficient)


def dice_loss_fcn(pred, target, smooth=1.0, layers_from=0):

    # Convert tensors as contiguous and drop any layer if required
    pred = pred.contiguous()[:, layers_from:, :, :]
    target = target.contiguous()[:, layers_from:, :, :]

    # Calculate intersection
    inter = (pred * target).sum(dim=2).sum(dim=2)

    # Calculate sum of areas
    area_sum = pred.sum(dim=2).sum(dim=2) + target.sum(dim=2).sum(dim=2)

    # Calculate dice coefficient
    dice = ((2. * inter + smooth) / (area_sum + smooth)).mean()

    # Return loss
    return 1 - dice


###############################################################################
# Function: loss_fcn
###############################################################################

# Custom loss function
# This function integrates both custom 'dice_loss_fcn' (see above) and standard
# Binary Cross Entropy Loss in one single loss function. By default the weight
# of each one is 0.5, but it can be modified.
# It also record the metrics, adding them up to the current ones.

# Inputs:
#     pred: (torch tensor) Predicted mask of shape (C x H x W)
#     target: (torch tensor) Target mask of same shape
#     metrics: (defaultdict) Dictionary. The key is a metric name and the
#         value is the current accumulated metric value (added in each batch)
#     w: (float) Weight. Ratio assigned to Binary Cross Entropy Loss. Range 0-1
#     mode: (string) It's just a label for the keys of metrics dict. It's
#         necessary to identify if the current metrics are from training or
#         from validation. Default: 'train'
#     lf: (integer) Layers from which the Dice coefficient is calculated

# Outputs:
#     loss: (float) Loss
#     metrics: (defaultdict) Updated metrics with the current batch metric
#         added.


def loss_fcn(pred, target, metrics, w=0.5, mode='train', lf=0):
    # w: weight of the binary cross entropy in the combined loss

    # Standard loss function
    bce = F.binary_cross_entropy_with_logits(pred, target)

    # Calculate prediction for dice comparison
    pred = torch.sigmoid(pred)

    # Dice metric loss
    dice_loss = dice_loss_fcn(pred, target, smooth=1.0, layers_from=lf)

    # Dice metric through torchmetrics
    # dice = dice_fcn(pred, target.type('torch.ShortTensor').to(DEVICE))

    # Combine both losses
    loss = bce * w + dice_loss * (1 - w)

    # Record accumulated metrics of the current epoch
    metrics[mode + '_bce'] += bce.data.cpu().numpy() * target.size(0)
    metrics[mode + '_dice'] += dice_loss.data.cpu().numpy() * target.size(0)
    metrics[mode + '_loss'] += loss.data.cpu().numpy() * target.size(0)
    metrics[mode + '_epoch_samples'] += target.size(0)

    return loss, metrics


###############################################################################
# Function: update_metrics
###############################################################################

# Update metrics at the end of one epoch. Transforms the sum of batch metrics
# in epoch metric.
# Metrics are calculated and added up in every batch. At the end of all the
# batches of an epoch, the metric epoch value has to be calculated through the
# mean of all the batch metrics. This is done in this function.

# See an important detail: metrics is a dict that has some keys that end with
# the string 'dice'. These items have a value that corresponds with dice loss
# at the input, but their value at the output is the dice coefficient.

# Inputs:
#     metrics: (defaultdict) Dictionary. The key is a metric name and the
#         value is the current accumulated metric value (the sum of all the
#         batches.

# Outputs:
#     metrics: (defaultdict) Updated metrics with the current epoch metrics.


def update_metrics(metrics):
    # Get the number os samples of train and validation or test
    train_epoch_samples = metrics['train_epoch_samples']
    valid_epoch_samples = metrics['valid_epoch_samples']
    test_epoch_samples = metrics['test_epoch_samples']

    # Delete these data from the metrics dictionary
    metrics.pop('train_epoch_samples')
    metrics.pop('valid_epoch_samples')
    metrics.pop('test_epoch_samples')

    # Calculate the average metrics of all the batches; i.e. the epoch metrics
    for m in metrics.keys():
        if m.startswith('train'):
            metrics[m] /= train_epoch_samples
        elif m.startswith('valid'):
            metrics[m] /= valid_epoch_samples
        elif m.startswith('test'):
            metrics[m] /= test_epoch_samples

        # If the metric is the dice loss, it needs to be inverted to calculate
        # the dice coefficient (i.e. 1 - dice_loss)
        if m.endswith('_dice'):
            metrics[m] = 1 - metrics[m]

    return metrics


###############################################################################
# Function: print_metrics
###############################################################################

# Helper function to print some results during training

# Inputs:
#     metrics: (defaultdict) Dictionary with metrics

# Outputs:
#     None. It displays training info.

def print_metrics(metrics, minimum=False):
    outputs = []

    if minimum:
        outputs.append("Tr loss: {:.4f} - Val loss: {:.4f} - Time: {:.1f}".format(
            metrics['train_loss'], metrics['valid_loss'], metrics['time']))
    else:
        for m in metrics.keys():
            outputs.append("{}: {:.4f}".format(m, metrics[m]))

    print("Epoch summary: {}".format(" - ".join(outputs)))


###############################################################################
# Function: train_model
###############################################################################

# Training loop

# Inputs:
#     model: (torch model) The model to be trained
#     optimizer: (optimizer) The optimizer object.
#     scheduler: (scheduler) The scheduler.
#     loaders: (dict) Dictionary with at least to items: loaders['train] and
#         loaders['valid'], contenining its corresponding dataloader.
#     num_epochs: (integer) Number of epochs to train
#     checkpoint: Optional. A checkpoint object as saved by the save_tr_model
#         function
#     verbose: (integer) Code to control the displayed info. 0: No info.
#     lf (integer): Layers from which the dice coefficient will be calculated.
#         Default is 0, menaing all layers will be used.

# Outputs:
#     model: (torch model) Trained model
#     training_log: (list) List of metrics in each epoch. Metrics is a dict.


def train_model(model, optimizer, scheduler, loaders, num_epochs=25,
                checkpoint={}, verbose=0, lf=0):

    # Training data retrieval if applicable
    if bool(checkpoint):
        training_log = checkpoint['training_log']
        valid_loss_min = np.min([x['valid_loss'] for x in training_log])
        trained_epochs = len(checkpoint['training_log'])
        best_state_dict = copy.deepcopy(model.state_dict())
    else:
        training_log = []
        valid_loss_min = np.Inf
        trained_epochs = 0
        best_state_dict = {}

    # Loop through epochs
    for epoch in range(1 + trained_epochs, 1 + trained_epochs + num_epochs):

        # Initialization of variables of the current epoch
        start_time = time.time()
        metrics = defaultdict(float)
        if verbose > 0:
            print(f"\nEpoch {epoch}")
            print("---------")

        # TRAINING
        model.train()

        # Loop in batches through the training data
        for inputs, labels in loaders['train']:
            # Tensors to gpu
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            # Reset gradients
            optimizer.zero_grad()
            # Forward pass
            outputs = model(inputs)
            # Calculate batch loss and update the metrics
            loss, metrics = loss_fcn(outputs, labels, metrics, mode='train',
                                     lf=lf)
            # Compute backpropagation
            loss.backward()
            # Update optimizer weights
            optimizer.step()

        # Apply new learning rate
        scheduler.step()

        # VALIDATION
        model.eval()

        # Loop in batches through the validation data
        for inputs, labels in loaders['valid']:
            # Tensors to gpu
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            # Forward pass without gradients
            with torch.no_grad(): outputs = model(inputs)
            # Calculate batch loss and update the metrics
            loss, metrics = loss_fcn(outputs, labels, metrics, mode='valid')

        # METRICS
        # Record time
        metrics['time'] = time.time() - start_time
        # Update metrics (calculate metrics per epoch)
        metrics = update_metrics(metrics)
        # Print metrics of the current epoch
        if verbose > 0:
            print_metrics(metrics, minimum=False)
        # Record metrics in training log
        training_log.append(metrics)

        # MODEL SAVE
        if metrics['valid_loss'] < valid_loss_min:
            if verbose > 0:
                print("saving best model")
            valid_loss_min = metrics['valid_loss']
            save_tr_model(model, optimizer, scheduler, training_log)
            best_state_dict = copy.deepcopy(model.state_dict())

    # load best model weights
    model.load_state_dict(best_state_dict)
    return model, training_log


###############################################################################
# Function: load_model
###############################################################################

# Load pretrained segmentation model from segmentation_models_pytorch and set
# some training parameters from optimizer and scheduler.

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

# Inputs:
#     option: (integer) Select the model number from the list above

# Outputs:
#     model: (torch model) Pretrained model without freezing any layer
#     arch: (string) Architecture name
#     encoder_name: (string) Encoder name
#     optimizer: (optimizer) Optimizer object
#     scheduler: (scheduler) Scheduler object


def load_model(option, num_class=3):

    if option == 1:
        arch = "DeepLabV3Plus"
        encoder_name = "efficientnet-b6"
        in_channels = 3
        encoder_weights = 'imagenet'
        out_channels = num_class

        model = smp.create_model(arch,
                                 encoder_name=encoder_name,
                                 encoder_weights=encoder_weights,
                                 in_channels=in_channels,
                                 classes=out_channels)

        model = model.to(DEVICE)
        optimizer = optim.Adam(filter(lambda param: param.requires_grad,
                                      model.parameters()), lr=0.001)
        scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    elif option == 2:
        arch = "MAnet"
        encoder_name = "timm-efficientnet-b6"
        encoder_weights = 'imagenet'
        in_channels = 3
        out_channels = num_class

        model = smp.create_model(arch,
                                 encoder_name=encoder_name,
                                 encoder_weights=encoder_weights,
                                 in_channels=in_channels,
                                 classes=out_channels)

        model = model.to(DEVICE)
        optimizer = optim.Adam(filter(lambda param: param.requires_grad,
                                      model.parameters()), lr=0.001)
        scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    elif option == 3:
        arch = "UnetPlusPlus"
        encoder_name = "efficientnet-b6"
        encoder_weights = 'imagenet'
        in_channels = 3
        out_channels = num_class

        model = smp.create_model(arch,
                                 encoder_name=encoder_name,
                                 encoder_weights=encoder_weights,
                                 in_channels=in_channels,
                                 classes=out_channels)

        model = model.to(DEVICE)
        optimizer = optim.Adam(filter(lambda param: param.requires_grad,
                                      model.parameters()), lr=0.001)
        scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    elif option == 4:
        arch = "Unet"
        encoder_name = "timm-efficientnet-b6"
        encoder_weights = 'imagenet'
        in_channels = 3
        out_channels = num_class

        model = smp.create_model(arch,
                                 encoder_name=encoder_name,
                                 encoder_weights=encoder_weights,
                                 in_channels=in_channels,
                                 classes=out_channels)

        model = model.to(DEVICE)
        optimizer = optim.Adam(filter(lambda param: param.requires_grad,
                                      model.parameters()), lr=0.001)
        scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    elif option == 5:
        arch = "U2Net"
        encoder_name = ""
        in_channels = 3
        out_channels = num_class

        model = U2NET(in_ch=in_channels,
                      out_ch=out_channels)

        model = model.to(DEVICE)
        optimizer = optim.Adam(filter(lambda param: param.requires_grad,
                                      model.parameters()), lr=0.001)
        scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    elif option == 6:
        arch = "PAN"
        encoder_name = "efficientnet-b6"
        encoder_weights = 'imagenet'
        in_channels = 3
        out_channels = num_class

        model = smp.create_model(arch,
                                 encoder_name=encoder_name,
                                 encoder_weights=encoder_weights,
                                 in_channels=in_channels,
                                 classes=out_channels)

        model = model.to(DEVICE)
        optimizer = optim.Adam(filter(lambda param: param.requires_grad,
                                      model.parameters()), lr=0.002)
        scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    elif option == 7:
        arch = "FPN"
        encoder_name = "timm-efficientnet-b6"
        encoder_weights = 'imagenet'
        in_channels = 3
        out_channels = num_class

        model = smp.create_model(arch,
                                 encoder_name=encoder_name,
                                 encoder_weights=encoder_weights,
                                 in_channels=in_channels,
                                 classes=out_channels)

        model = model.to(DEVICE)
        optimizer = optim.Adam(filter(lambda param: param.requires_grad,
                                      model.parameters()), lr=0.001)
        scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    elif option == 8:
        arch = "PSPNet"
        encoder_name = "timm-efficientnet-b6"
        encoder_weights = 'imagenet'
        in_channels = 3
        out_channels = num_class

        model = smp.create_model(arch,
                                 encoder_name=encoder_name,
                                 encoder_weights=encoder_weights,
                                 in_channels=in_channels,
                                 classes=out_channels)

        model = model.to(DEVICE)
        optimizer = optim.Adam(filter(lambda param: param.requires_grad,
                                      model.parameters()), lr=0.002)
        scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    elif option == 9:
        arch = "Linknet"
        encoder_name = "timm-efficientnet-b6"
        encoder_weights = 'imagenet'
        in_channels = 3
        out_channels = num_class

        model = smp.create_model(arch,
                                 encoder_name=encoder_name,
                                 encoder_weights=encoder_weights,
                                 in_channels=in_channels,
                                 classes=out_channels)

        model = model.to(DEVICE)
        optimizer = optim.Adam(filter(lambda param: param.requires_grad,
                                      model.parameters()), lr=0.001)
        scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    return model, arch, encoder_name, optimizer, scheduler


###############################################################################
# Function: plot_training_log
###############################################################################

# Helper function to plot the training log

# Inputs:
#     tr_log: (list) The training log is a list of metrics. Each item if the
#         list is a dict of the metrics of that epoch. The dict contains at
#         least the keys 'train_loss', 'valid_loss', 'train_dice' and
#         'valid_dice'

# Outputs:
#     None - It displays the plot


def plot_training_log(tr_log):
    losses = np.array([[m['train_loss'], m['valid_loss']] for m in tr_log])
    tr_losses, vl_losses = losses[:, 0], losses[:, 1]
    x = range(1, 1 + len(losses))
    tr_max, tr_min = np.max(tr_losses), np.min(tr_losses)
    epoch_min = 1 + np.argmin(vl_losses)
    val_min = np.min(vl_losses)

    metrics = np.array([[m['train_dice'], m['valid_dice']] for m in tr_log])
    tr_dice, vl_dice = metrics[:, 0], metrics[:, 1]
    # trm_max, trm_min = np.max(tr_dice), np.min(tr_dice)
    epoch_m_max = 1 + np.argmax(vl_dice)
    dice_max = np.max(vl_dice)

    plt.style.use("classic")
    fig = plt.figure(figsize=(14, 5))
    plt.suptitle('Training log', fontsize=16)

    # Subplot 1 - Losses
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.set_title('Losses')
    ax1.plot(x, tr_losses, label='training loss')
    ax1.plot(x, vl_losses, label='validation loss')
    plt.xlabel('epochs')
    plt.ylabel('loss')
    ax1.legend()
    ax1.annotate('valid min: {:.4f}'.format(val_min), xy=(epoch_min, val_min),
                 xytext=(round(0.5 * len(tr_losses)), 3*(tr_max - tr_min)/4 +
                         tr_min), arrowprops=dict(facecolor='black',
                                                  shrink=0.05))
    plt.xlim(0, len(tr_losses))

    # Subplot 2 - Metric
    ax2 = fig.add_subplot(1, 2, 2)
    ax2.set_title('Dice metric')
    ax2.plot(x, tr_dice, label='training dice')
    ax2.plot(x, vl_dice, label='validation dice')
    plt.xlabel('epochs')
    plt.ylabel('dice metric')
    ax2.legend(loc='lower right')
    ax2.annotate('max dice: {:.4f}'.format(dice_max), xy=(epoch_m_max, dice_max),
                 xytext=(round(0.5 * len(tr_losses)), 0.5 +
                         tr_min), arrowprops=dict(facecolor='black',
                                                  shrink=0.05))
    plt.xlim(0, len(tr_losses))
    plt.ylim(0, 1)


###############################################################################
# Function: show_compare
###############################################################################

# Helper function to show the original image, the original mask and the
# predicted mask

# Inputs:
#     img: (np-array) Original image as numpy array
#     mask: (np-array) Original (flat) mask as numpy array
#     pred: (np-array) Predicted (flat) mask as numpy array

# Outputs:
#     None - It displays the plot


# Helper function to show the image, the mask and the predicted mask
def show_compare(img, mask, pred):
    # plt.style.use("ggplot")
    plt.style.use("classic")
    fig = plt.figure(figsize=(14, 5))
    plt.suptitle('Image, mask and prediction', fontsize=16)

    # Subplot 1 - Image
    ax1 = fig.add_subplot(1, 3, 1)
    ax1.set_title('Image')
    plt.imshow(img)

    # Subplot 2 - Mask
    ax1 = fig.add_subplot(1, 3, 2)
    ax1.set_title('Mask')
    plt.imshow(mask, cmap='gray')

    # Subplot 3 - Prediction
    ax1 = fig.add_subplot(1, 3, 3)
    ax1.set_title('Prediction')
    plt.imshow(pred, cmap='gray')

    plt.show()


###############################################################################
# Function: test
###############################################################################

# Helper function to run the model with test images and extract the bundled
# metrics of the test data.

# Inputs:
#     loader: (dataloader) Test dataloader
#     model: (model) Trained model

# Outputs:
#     metrics: (dict) Dictionary with metrics


def test(loader, model):

    # Prepare model and parameters
    model.eval()
    model.to(DEVICE)
    metrics = defaultdict(float)

    # Loop in batches through the test data
    for inputs, labels in loader:
        # Tensors to gpu
        inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
        # Forward pass without gradients
        with torch.no_grad():
            outputs = model(inputs)
        # Calculate batch loss and update the accumulated metrics
        _, metrics = loss_fcn(outputs, labels, metrics, mode='test')

    # Update metrics (calculate bundled metrics)
    metrics = update_metrics(metrics)

    return metrics


###############################################################################
# Function: extract_features
###############################################################################

# This function extract a set of image and mask features and return a dict with
# them.

# Inputs:
#     im: (path or PIL image) Path to image or PIL grayscale image
#     mk: (path or PIL image) Path to mask or PIL graysacale mask
#     extr_nuc: (extractor) Object featureextractor.RadiomicsFeatureExtractor.
#         The extractor of the nucleus features.
#     extr_cyt: (extractor) Object featureextractor.RadiomicsFeatureExtractor.
#         The extractor of the cytoplasm features.
#     cell_type: (string) Optional. Cell type (i.e. lymphocyte, monocyte, ...)

# Outputs:
#     d: (dict) Dictionary with all extracted features (the prefix 'nuc'
#         stands for 'nucleus', 'cyt' stands for 'cytoplasm' and 'mix'
#         stands for 'mix', a combination of both):
#               cell_type
#               image_path
#               mask_path
#               nuc_Elongation
#               nuc_MajorAxisLength
#               nuc_MaximumDiameter
#               nuc_MeshSurface
#               nuc_MinorAxisLength
#               nuc_Perimeter
#               nuc_PerimeterSurfaceRatio
#               nuc_PixelSurface
#               nuc_Sphericity
#               nuc_10Percentile
#               nuc_90Percentile
#               nuc_Energy
#               nuc_Entropy
#               nuc_InterquartileRange
#               nuc_Kurtosis
#               nuc_Maximum
#               nuc_MeanAbsoluteDeviation
#               nuc_Mean
#               nuc_Median
#               nuc_Minimum
#               nuc_Range
#               nuc_RobustMeanAbsoluteDeviation
#               nuc_RootMeanSquared
#               nuc_Skewness
#               nuc_TotalEnergy
#               nuc_Uniformity
#               nuc_Variance
#               cyt_Elongation
#               cyt_MajorAxisLength
#               cyt_MaximumDiameter
#               cyt_MeshSurface
#               cyt_MinorAxisLength
#               cyt_Perimeter
#               cyt_PerimeterSurfaceRatio
#               cyt_PixelSurface
#               cyt_Sphericity
#               cyt_10Percentile
#               cyt_90Percentile
#               cyt_Energy
#               cyt_Entropy
#               cyt_InterquartileRange
#               cyt_Kurtosis
#               cyt_Maximum
#               cyt_MeanAbsoluteDeviation
#               cyt_Mean
#               cyt_Median
#               cyt_Minimum
#               cyt_Range
#               cyt_RobustMeanAbsoluteDeviation
#               cyt_RootMeanSquared
#               cyt_Skewness
#               cyt_TotalEnergy
#               cyt_Uniformity
#               cyt_Variance
#               mix_Elongation
#               mix_MajorAxisLength
#               mix_MaximumDiameter
#               mix_MeshSurface
#               mix_MinorAxisLength
#               mix_Perimeter
#               mix_PerimeterSurfaceRatio
#               mix_PixelSurface
#               mix_Sphericity


def extract_features(im, mk, extr_nuc, extr_cyt, cell_type='unknown'):

    # Check if inputs are paths or PIL images
    if isinstance(im, str):
        # 'im' is a path
        im_path = im
        image = sitk.GetImageFromArray(Image.open(im_path).convert('L'))
    else:
        # 'im' is a PIL grayscale image
        im_path = ""
        image = sitk.GetImageFromArray(im)
    if isinstance(mk, str):
        # 'mk' is a path
        mk_path = mk
        mask = sitk.GetImageFromArray(Image.open(mk_path))
    else:
        # 'mk' is a PIL grayscale image
        mk_path = ""
        mask = sitk.GetImageFromArray(mk)

    # Extract features
    feat_nuc = extr_nuc.execute(image, mask)  # nucleus
    feat_cyt = extr_cyt.execute(image, mask)  # cytoplasm

    # Create a dictionary of features
    d0 = {}
    d0['cell_type'] = cell_type
    d0['image_path'] = im_path
    d0['mask_path'] = mk_path

    # Loop through each key and extract values of nucleus and cytoplasm
    d1, d2 = {}, {}
    for key, value in six.iteritems(feat_nuc):
        # Get feature values from both extractors (nucleus and cytoplasm)
        if key.startswith("original_"):
            new_key = key.split('_')[-1]
            d1['nuc_' + new_key] = float(value)
            d2['cyt_' + new_key] = float(feat_cyt[key])

    # Add shape relationship between nucleus and cytoplasm
    d = {**d0, **d1, **d2}
    d['mix_Elongation'] = d1['nuc_Elongation'] / d2['cyt_Elongation']
    d['mix_MajorAxisLength'] = d1['nuc_MajorAxisLength'] / d2['cyt_MajorAxisLength']
    d['mix_MaximumDiameter'] = d1['nuc_MaximumDiameter'] / d2['cyt_MaximumDiameter']
    d['mix_MeshSurface'] = d1['nuc_MeshSurface'] / (d1['nuc_MeshSurface'] + d2['cyt_MeshSurface'])
    d['mix_MinorAxisLength'] = d1['nuc_MinorAxisLength'] / d2['cyt_MinorAxisLength']
    d['mix_Perimeter'] = d1['nuc_Perimeter'] / d2['cyt_Perimeter']
    d['mix_PerimeterSurfaceRatio'] = d1['nuc_PerimeterSurfaceRatio'] / d2['cyt_PerimeterSurfaceRatio']
    d['mix_PixelSurface'] = d1['nuc_PixelSurface'] / (d1['nuc_PixelSurface'] + d2['cyt_PixelSurface'])
    d['mix_Sphericity'] = d1['nuc_Sphericity'] / d2['cyt_Sphericity']

    return d
