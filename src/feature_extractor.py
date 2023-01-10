# -*- coding: utf-8 -*-

# This script extracts the features of all the images and masks present in a
# folder, and saves them into .csv files.

# Select the folder of the dataset in the section:
# "Global variables and parameters"

# This script should be run before you start training your model, so you can
# have the .csv files in order to do the posterior analysis.


###############################################################################
# Importation of libraries
###############################################################################

import os
from imutils import paths
import pathlib
import sys
from radiomics import featureextractor
import pandas as pd

PROJECT_DIR = str(pathlib.Path().resolve())
sys.path.append(PROJECT_DIR)
from custom_utils import extract_features

###############################################################################
# Global variables and parameters
###############################################################################

# Paths to the datasets
ds = 'data'
IMAGE_PATH = os.path.join(PROJECT_DIR, ds + '/img')
MASK_PATH = os.path.join(PROJECT_DIR, ds + '/msk')


###############################################################################
# Loading datasets
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


###############################################################################
# Labels of images depending on cell type
###############################################################################

# Define cells to look for. Folders should have these names
cells = ('monocyte', 'lymphocyte', 'HCL', 'APL')

# Extract label for each image depending on path
labels = []
for path in image_paths:
    for cell in cells:
        if cell in path:
            labels.append(cell)
            break


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
# Feature extractor loop and save results
###############################################################################

# Loop through the cell types
df = pd.DataFrame()

# Loop through the files
for im, mk, cell_type in zip(image_paths, mask_paths, labels):
    # Extract features as a dict
    d = extract_features(im, mk, extr_nuc, extr_cyt, cell_type)
    # Add features to the dataframe
    df = df.append(d, ignore_index=True)

# Save dataframe to a .csv file
filename = 'df.csv'
df.to_csv(os.path.join(PROJECT_DIR, "pyrad_data", filename), sep='\t')

# Create partial dataframes depending on cell type
monocyte_df = df.loc[df["cell_type"] == cells[0]]
lymphocyte_df = df.loc[df["cell_type"] == cells[1]]
hcl_df = df.loc[df["cell_type"] == cells[2]]
apl_df = df.loc[df["cell_type"] == cells[3]]

# Get only numerical values and calculate means
fnum = 3
monocyte_dfs = monocyte_df.iloc[:, fnum:].mean()
monocyte_dfs = pd.DataFrame(dict(monocyte_dfs), index=[0])

lymphocyte_dfs = lymphocyte_df.iloc[:, fnum:].mean()
lymphocyte_dfs = pd.DataFrame(dict(lymphocyte_dfs), index=[0])

hcl_dfs = hcl_df.iloc[:, fnum:].mean()
hcl_dfs = pd.DataFrame(dict(hcl_dfs), index=[0])

apl_dfs = apl_df.iloc[:, fnum:].mean()
apl_dfs = pd.DataFrame(dict(apl_dfs), index=[0])

# Save results
name = "monocyte_df.csv"
fpath = os.path.join(PROJECT_DIR, "pyrad_data", name)
monocyte_df.to_csv(fpath, sep='\t')

name = "lymphocyte_df.csv"
fpath = os.path.join(PROJECT_DIR, "pyrad_data", name)
lymphocyte_df.to_csv(fpath, sep='\t')

name = "hcl_df.csv"
fpath = os.path.join(PROJECT_DIR, "pyrad_data", name)
hcl_df.to_csv(fpath, sep='\t')

name = "apl_df.csv"
fpath = os.path.join(PROJECT_DIR, "pyrad_data", name)
apl_df.to_csv(fpath, sep='\t')

# Save summary dataframes
name = "monocyte_df_summary.csv"
fpath = os.path.join(PROJECT_DIR, "pyrad_data", name)
monocyte_dfs.to_csv(fpath, sep='\t')

name = "lymphocyte_df_summary.csv"
fpath = os.path.join(PROJECT_DIR, "pyrad_data", name)
lymphocyte_dfs.to_csv(fpath, sep='\t')

name = "hcl_df_summary.csv"
fpath = os.path.join(PROJECT_DIR, "pyrad_data", name)
hcl_dfs.to_csv(fpath, sep='\t')

name = "apl_df_summary.csv"
fpath = os.path.join(PROJECT_DIR, "pyrad_data", name)
apl_dfs.to_csv(fpath, sep='\t')
