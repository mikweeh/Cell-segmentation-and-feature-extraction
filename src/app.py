# -*- coding: utf-8 -*-

if __name__ == "__main__":

    ###########################################################################
    # Libraries and global variables
    ###########################################################################

    import gradio as gr
    import numpy as np
    import torch
    from torchvision.transforms.functional import to_tensor
    import albumentations as A
    import pathlib
    import sys
    from radiomics import featureextractor
    import os
    import pandas as pd
    from PIL import Image
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker
    import matplotlib as mpl
    mpl.use('agg')

    # Remember to connect from your web browser to
    # http://localhost:7860/

    PROJECT_DIR = str(pathlib.Path().resolve())
    sys.path.append(PROJECT_DIR)
    import custom_utils as h

    FILENAME = 'unetpp_2ldice.pth'
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    IM_SIZE = 224
    transform_test = A.Compose([A.CenterCrop(width=IM_SIZE, height=IM_SIZE)])


    ###########################################################################
    # Load the segmentation model
    ###########################################################################

    # Model option
    m_option = 3

    # Load the model
    model, _, _, _, _ = h.load_model(m_option, 3)

    # Load checkpoint (the dict saved by 'save_tr_model')
    checkpoint = torch.load(FILENAME, map_location=torch.device('cpu'))

    # Pass the current model to cpu
    model = model.to('cpu')

    # Load the state_dict of the model
    model.load_state_dict(checkpoint['model_state_dict'])

    # Pass the model to DEVICE
    model.eval()
    model = model.to(DEVICE)


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
    # Load dataframes of extracted features
    ###############################################################################

    def df_summary(df):
        df2 = df.quantile(q=0.02).to_frame().T
        df98 = df.quantile(q=0.98).T.to_frame().T
        dfs = df.describe().append(pd.concat([df2, df98],
                                            ignore_index=True))
        dfs.index = ["count", "mean", "std", "min", "25%", "50%", "75%",
                    "max", "2%", "98%"]
        # Correction of magnitude order in energy (divided by 1e6)
        dfs.loc[:, ['nuc_Energy', 'cyt_Energy']] = dfs.loc[:, ['nuc_Energy', \
            'cyt_Energy']] / 1e6

        return dfs


    # Monocytes
    name = "monocyte_df.csv"
    fp = os.path.join(PROJECT_DIR, "pyrad_data", name)
    df = pd.read_csv(fp, sep="\t").iloc[:, 4:]
    monocyte_dfs = df_summary(df)

    # Lymphocytes
    name = "lymphocyte_df.csv"
    fp = os.path.join(PROJECT_DIR, "pyrad_data", name)
    df = pd.read_csv(fp, sep="\t").iloc[:, 4:]
    lymphocyte_dfs = df_summary(df)

    # HCL
    name = "hcl_df.csv"
    fp = os.path.join(PROJECT_DIR, "pyrad_data", name)
    df = pd.read_csv(fp, sep="\t").iloc[:, 4:]
    hcl_dfs = df_summary(df)

    # APL
    name = "apl_df.csv"
    fp = os.path.join(PROJECT_DIR, "pyrad_data", name)
    df = pd.read_csv(fp, sep="\t").iloc[:, 4:]
    apl_dfs = df_summary(df)

    # Feature tags
    name = "feature_tags.csv"
    fp = os.path.join(PROJECT_DIR, "pyrad_data", name)
    feature_tags = pd.read_csv(fp, sep="\t")
    current_features = pd.DataFrame([])


    ###############################################################################
    # Set up some graphic features
    ###############################################################################

    # Matplotlib features
    t = 70
    plt.rc('font', size=t)  # controls default text size
    plt.rc('axes', titlesize=t)  # fontsize of the title
    plt.rc('axes', labelsize=t)  # fontsize of the x and y labels
    plt.rc('xtick', labelsize=t)  # fontsize of the x tick labels
    plt.rc('ytick', labelsize=t)  # fontsize of the y tick labels
    plt.rc('legend', fontsize=t)  # fontsize of the legend
    mpl.rc('figure', max_open_warning=0)

    # Feature indexes (grouping depending on magnitude order)
    shape1 = [0, 6, 8, 27, 33, 35, 54, 55, 56, 58, 59, 60, 61, 62]
    shape2 = [1, 2, 4, 5, 7, 28, 29, 31, 32, 34]
    f_ord1 = [9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 25,
              26, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50,
              52, 53]


    ###############################################################################
    # Generate static figures and dataframes to show in general plots
    ###############################################################################

    # Function to generate static data
    def static_data(dfs1, dfs2, pos, cell1='monocyte', cell2='lymphocyte',
                    location='lower right'):
        # Static plot and dataframe
        # Extract from the dataframes 'dfs1' and 'dfs2' the data to plot and also
        # generate the dataframes of the general information
        dt1 = dfs1.iloc[:, pos].copy()
        dt2 = dfs2.iloc[:, pos].copy()
        ixs = np.arange(0, dt1.shape[1])

        # Dataframe1 with the table to show
        df1 = pd.DataFrame(ixs, columns=['Index'])
        df1['Features'] = list(feature_tags.iloc[pos, 0])
        df1['Percentile 2'] = dt1.loc["2%"].values
        df1['Percentile 50'] = dt1.loc["50%"].values
        df1['Percentile 98'] = dt1.loc["98%"].values
        df1['Mean'] = dt1.loc["mean"].values
        df1['Std.Deviation'] = dt1.loc["std"].values

        # Dataframe2 with the table to show
        df2 = pd.DataFrame(ixs, columns=['Index'])
        df2['Features'] = list(feature_tags.iloc[pos, 0])
        df2['Percentile 2'] = dt2.loc["2%"].values
        df2['Percentile 50'] = dt2.loc["50%"].values
        df2['Percentile 98'] = dt2.loc["98%"].values
        df2['Mean'] = dt2.loc["mean"].values
        df2['Std.Deviation'] = dt2.loc["std"].values

        color = 'blue'
        fig, ax = plt.subplots(1, figsize=(90, 30))

        # Adapt scale in case we are dealing with first order pixel-value features
        if pos[0] == 9:  # 9 is the value of the first index in pixel-val features
            # Arrange plot of first order features
            dt1.iloc[:, 31] = dt1.iloc[:, 31] + 1  # Correction of skewness plot
            dt2.iloc[:, 31] = dt2.iloc[:, 31] + 1  # Correction of skewness plot

        # Quantiles dt1
        plt.plot(dt1.loc[["2%", "98%"], :].T, 'm--',
                 label=["percentile 2", "percentile 98"], linewidth=1)
        plt.fill_between(ixs, dt1.loc["2%", :], dt1.loc["98%", :],
                         facecolor='gray', alpha=0.2)
        # Mean dt1
        plt.plot(dt1.loc["mean", :].T, 'm--', label=f"{cell1} means", linewidth=5)

        # Quantiles dt2
        plt.plot(dt2.loc[["2%", "98%"], :].T, 'm--', linewidth=1)
        plt.fill_between(ixs, dt2.loc["2%", :], dt2.loc["98%", :],
                         facecolor='orange', alpha=0.2)
        # Mean dt2
        plt.plot(dt2.loc["mean", :].T, 'r-', label=f"{cell2} means", linewidth=5)

        # Other parameters
        plt.ylabel('feature value', color=color)
        plt.xlabel('feature index', color=color)
        ax.set_yscale('log')
        ax.xaxis.set_major_locator(mticker.FixedLocator(ixs))
        ax.set_xticklabels(labels=list(ixs))
        ax.legend(loc=location)
        ax.grid(True)
        ax.grid(color='b', alpha=0.5, linestyle='dashed', linewidth=1)
        ax.set_title(f'Features of {cell1}s and {cell2}s', color=color)
        ax.tick_params(axis='x', colors=color, rotation=0)
        ax.tick_params(axis='y', colors=color)

        return fig, df1, df2


    ###############################################################################
    # Comparison between monocytes and lymphocytes
    ###############################################################################

    # Shape features 1
    fig01, df_s1_mon, df_s1_lym = static_data(monocyte_dfs, lymphocyte_dfs, shape1,
                                            'monocyte', 'lymphocyte', 'upper left')
    # Shape features 2
    fig02, df_s2_mon, df_s2_lym = static_data(monocyte_dfs, lymphocyte_dfs, shape2,
                                            'monocyte', 'lymphocyte', 'upper left')

    # Pixel-value features
    fig03, df_pv_mon, df_pv_lym = static_data(monocyte_dfs, lymphocyte_dfs, f_ord1,
                                            'monocyte', 'lymphocyte', 'lower left')


    ###############################################################################
    # Comparison between monocytes and HCL
    ###############################################################################

    # Shape features 1
    fig04, _, df_s1_hcl = static_data(monocyte_dfs, hcl_dfs, shape1,
                                    'monocyte', 'HCL', 'upper left')
    # Shape features 2
    fig05, _, df_s2_hcl = static_data(monocyte_dfs, hcl_dfs, shape2,
                                    'monocyte', 'HCL', 'upper left')

    # Pixel-value features
    fig06, _, df_pv_hcl = static_data(monocyte_dfs, hcl_dfs, f_ord1,
                                    'monocyte', 'HCL', 'lower left')


    ###############################################################################
    # Comparison between monocytes and APL
    ###############################################################################

    # Shape features 1
    fig07, _, df_s1_apl = static_data(monocyte_dfs, apl_dfs, shape1,
                                    'monocyte', 'APL', 'upper left')
    # Shape features 2
    fig08, _, df_s2_apl = static_data(monocyte_dfs, apl_dfs, shape2,
                                    'monocyte', 'APL', 'upper left')

    # Pixel-value features
    fig09, _, df_pv_apl = static_data(monocyte_dfs, apl_dfs, f_ord1,
                                    'monocyte', 'APL', 'lower left')


    ###############################################################################
    # Comparison between lymphocytes and HCL
    ###############################################################################

    # Shape features 1
    fig10, _, _ = static_data(lymphocyte_dfs, hcl_dfs, shape1,
                            'lymphocyte', 'HCL', 'upper left')
    # Shape features 2
    fig11, _, _ = static_data(lymphocyte_dfs, hcl_dfs, shape2,
                            'lymphocyte', 'HCL', 'upper left')

    # Pixel-value features
    fig12, _, _ = static_data(lymphocyte_dfs, hcl_dfs, f_ord1,
                            'lymphocyte', 'HCL', 'upper left')


    ###############################################################################
    # Comparison between lymphocytes and APL
    ###############################################################################

    # Shape features 1
    fig13, _, _ = static_data(lymphocyte_dfs, apl_dfs, shape1,
                            'lymphocyte', 'APL', 'upper left')
    # Shape features 2
    fig14, _, _ = static_data(lymphocyte_dfs, apl_dfs, shape2,
                            'lymphocyte', 'APL', 'upper left')

    # Pixel-value features
    fig15, _, _ = static_data(lymphocyte_dfs, apl_dfs, f_ord1,
                            'lymphocyte', 'APL', 'upper left')


    ###############################################################################
    # Comparison between HCL and APL
    ###############################################################################

    # Shape features 1
    fig16, _, _ = static_data(hcl_dfs, apl_dfs, shape1,
                            'HCL', 'APL', 'upper left')
    # Shape features 2
    fig17, _, _ = static_data(hcl_dfs, apl_dfs, shape2,
                            'HCL', 'APL', 'upper left')

    # Pixel-value features
    fig18, _, _ = static_data(hcl_dfs, apl_dfs, f_ord1,
                            'HCL', 'APL', 'lower left')


    ###############################################################################
    # Information text
    ###############################################################################

    info_text = """
    _All features extracted with pyradiomics. See details in : https://pyradiomics.readthedocs.io/en/latest/features.html_

    ## Elongation
    Elongation shows the relationship between the two largest principal components in the ROI shape.
    For computational reasons, this feature is defined as the inverse of true elongation.

    ## MajorAxisLength
    This feature yield the largest axis length of the ROI-enclosing ellipsoid and is calculated using the largest
    principal component.

    ## MaximumDiameter
    Maximum diameter is defined as the largest pairwise Euclidean distance between ROI surface mesh vertices.

    ## MinorAxisLength
    This feature yield the second-largest axis length of the ROI-enclosing ellipsoid and is calculated using the largest
    principal component.

    ## Perimeter
    Perimeter of ROI.

    ## PerimeterSurfaceRatio
    Here, a lower value indicates a more compact (circle-like) shape.

    ## PixelSurface
    The surface area of the ROI.

    ## Sphericity
    Sphericity is the ratio of the perimeter of the ROI to the perimeter of a circle with
    the same surface area as the ROI.

    ## 10Percentile
    10th percentile.

    ## 90Percentile
    90th percentile.

    ## Energy
    Energy is a measure of the magnitude of pixel values in an image. A larger values implies a greater sum of the
    squares of these values. The values showed are divided by 1e6 to avoid the difference of order
    of magnitude with the rest of features.

    ## Entropy
    Entropy specifies the uncertainty/randomness in the image values. It measures the average amount of information
    required to encode the image values.

    ## InterquartileRange
    Interquartile range between 25th and 75th percentiles.

    ## Kurtosis
    Kurtosis is a measure of the 'peakedness' of the distribution of values in the image ROI.

    ## Maximum
    The maximum gray level intensity within the ROI.

    ## MeanAbsoluteDeviation
    Mean Absolute Deviation is the mean distance of all intensity values from the Mean Value of the image array.

    ## Mean
    The average gray level intensity within the ROI.

    ## Median
    The median gray level intensity within the ROI.

    ## Minimum
    The minimum gray level intensity within the ROI.

    ## Range
    The range of gray values in the ROI.

    ## RobustMeanAbsoluteDeviation
    Robust Mean Absolute Deviation is the mean distance of all intensity values
    from the Mean Value calculated on the subset of image array with gray levels in between, or equal
    to the 10th and 90th percentile.

    ## RootMeanSquared
    RMS is the square-root of the mean of all the squared intensity values. It is another measure of the magnitude of
    the image values.

    ## Skewness
    Skewness measures the asymmetry of the distribution of values about the Mean value. Depending on where the tail is
    elongated and the mass of the distribution is concentrated, this value can be positive or negative. The value
    represented in the plot, as it is in logarithmic scale, is the real value plus one, to avoid that negative values
    become minus infinite.

    ## Uniformity
    Uniformity is a measure of the sum of the squares of each intensity value. This is a measure of the homogeneity of
    the image array, where a greater uniformity implies a greater homogeneity or a smaller range of discrete intensity
    values.

    ## Variance
    Variance is the the mean of the squared distances of each intensity value from the Mean value. This is a measure of
    the spread of the distribution about the mean.

    ## Combined elongation
    Nucleus elongation divided by cytoplasm elongation.

    ## Combined MajorAxisLength
    Nucleus major axis length divided by cytoplasm major axis length.

    ## Combined MaximumDiameter
    Nucleus maximum diameter divided by cytoplasm maximum diameter.

    ## Combined MinorAxisLength
    Nucleus minor axis length divided by cytoplasm minor axis length.

    ## Combined Perimeter
    Nucleus perimeter divided by cytoplasm perimeter.

    ## Combined PerimeterSurfaceRatio
    Nucleus perimeter surface ratio divided by cytoplasm perimeter surface ratio.

    ## Combined PixelSurface
    Nucleus pixel surface divided by the sum of nucleus and cytoplasm pixel surfaces.

    ## Combined Sphericity
    Nucleus sphericity divided by cytoplasm sphericity.
    """


    ###############################################################################
    # Functions to coordinate the graphical usen interface with Gradio and the
    # data flow
    ###############################################################################

    # Direct execution of forward pass in the model and feature extraction
    def create_mask(input_img):
        # Create mask, extract features and leave them in global 'current_features'

        global current_features

        # Apply albumentations transforms
        augmented = transform_test(image=input_img)
        image = augmented['image']

        # Convert into tensor and pass it to DEVICE with batch dimension
        tensor_im = torch.unsqueeze(to_tensor(image).to(DEVICE), dim=0)

        # Generate predicted mask
        deep_mask = torch.sigmoid(model(tensor_im)).to('cpu')

        # Transform predicted deep mask into flat mask
        mask = h.flat_mask(torch.squeeze(deep_mask), level=0.5)

        # Convert to PIL image to extract with pyradiomics
        PIL_im = Image.fromarray(image).convert('L')
        PIL_mk = Image.fromarray(mask)

        # Extract features as global variable
        current_features = h.extract_features(PIL_im, PIL_mk, extr_nuc, extr_cyt)
        current_features['nuc_Energy'] = current_features['nuc_Energy'] / 1e6
        current_features['cyt_Energy'] = current_features['cyt_Energy'] / 1e6
        del current_features['cell_type']
        del current_features['image_path'],
        del current_features['mask_path']

        return mask


    # Instanciate figures of monocyte features
    fig19, _ = plt.subplots(1, figsize=(90, 30))
    fig20, _ = plt.subplots(1, figsize=(90, 30))
    fig21, _ = plt.subplots(1, figsize=(90, 30))

    # Instanciate figures of lymphocyte features
    fig22, _ = plt.subplots(1, figsize=(90, 30))
    fig23, _ = plt.subplots(1, figsize=(90, 30))
    fig24, _ = plt.subplots(1, figsize=(90, 30))

    # Instanciate figures of HCL features
    fig25, _ = plt.subplots(1, figsize=(90, 30))
    fig26, _ = plt.subplots(1, figsize=(90, 30))
    fig27, _ = plt.subplots(1, figsize=(90, 30))

    # Instanciate figures of APL features
    fig28, _ = plt.subplots(1, figsize=(90, 30))
    fig29, _ = plt.subplots(1, figsize=(90, 30))
    fig30, _ = plt.subplots(1, figsize=(90, 30))


    def close_figs():
        global fig19, fig20, fig21, fig22, fig23, fig24
        global fig25, fig26, fig27, fig28, fig29, fig30

        plt.close(fig19.number)
        plt.close(fig20.number)
        plt.close(fig21.number)
        plt.close(fig22.number)
        plt.close(fig23.number)
        plt.close(fig24.number)
        plt.close(fig25.number)
        plt.close(fig26.number)
        plt.close(fig27.number)
        plt.close(fig28.number)
        plt.close(fig29.number)
        plt.close(fig30.number)


    # Coordination of Gradio controls and dataframes
    def calc_features():

        global fig19, fig20, fig21, fig22, fig23, fig24
        global fig25, fig26, fig27, fig28, fig29, fig30

        # Close all dinamic figures to release memory
        close_figs()

        # Assign figures to global variables
        fig19, df19, fig20, df20, fig21, df21 = feat_fcn('monocyte')
        fig22, df22, fig23, df23, fig24, df24 = feat_fcn('lymphocyte')
        fig25, df25, fig26, df26, fig27, df27 = feat_fcn('hcl')
        fig28, df28, fig29, df29, fig30, df30 = feat_fcn('apl')

        return [fig19, df19, fig20, df20, fig21, df21,
                fig22, df22, fig23, df23, fig24, df24,
                fig25, df25, fig26, df26, fig27, df27,
                fig28, df28, fig29, df29, fig30, df30]


    # Refresh functions
    def update1():
        global fig19, fig20, fig21

        # Close figures to release memory
        plt.close(fig19.number)
        plt.close(fig20.number)
        plt.close(fig21.number)

        # Create new figures and dataframes
        cell = 'monocyte'
        fig19, df19, fig20, df20, fig21, df21 = feat_fcn(cell)
        return [fig19, df19, fig20, df20, fig21, df21]


    def update2():
        global fig22, fig23, fig24

        # Close figures to release memory
        plt.close(fig22.number)
        plt.close(fig23.number)
        plt.close(fig24.number)

        # Create new figures and dataframes
        cell = 'lymphocyte'
        fig22, df22, fig23, df23, fig24, df24 = feat_fcn(cell)
        return [fig22, df22, fig23, df23, fig24, df24]


    def update3():
        global fig25, fig26, fig27

        # Close figures to release memory
        plt.close(fig25.number)
        plt.close(fig26.number)
        plt.close(fig27.number)

        # Create new figures and dataframes
        cell = 'hcl'
        fig25, df25, fig26, df26, fig27, df27 = feat_fcn(cell)
        return [fig25, df25, fig26, df26, fig27, df27]


    def update4():
        global fig28, fig29, fig30

        # Close figures to release memory
        plt.close(fig28.number)
        plt.close(fig29.number)
        plt.close(fig30.number)

        # Create new figures and dataframes
        cell = 'apl'
        fig28, df28, fig29, df29, fig30, df30 = feat_fcn(cell)
        return [fig28, df28, fig29, df29, fig30, df30]


    # Generation of dataframe in the current cell features comparison
    def make_df(data, pos):
        # Arrange dataframe and plotable data of selected features
        dt = data.iloc[:, pos].copy()
        ixs = np.arange(0, dt.shape[1])
        df = pd.DataFrame(ixs, columns=['Index'])
        df['Features'] = list(feature_tags.iloc[pos, 0])
        # df['Current Values'] = np.array(list(current_features.values()))[pos]
        df['Current Values'] = np.array(list(data.loc["current"]))[pos]
        # df['Mean Cell Values'] = np.array(list(data.values[1]))[pos]  # Means
        df['Mean Cell Values'] = np.array(list(data.loc["mean"]))[pos]

        return df, dt


    # Generation of figures in the current cell features comparison
    def plot_arrange(data, cell, position='lower right'):
        # Arrange plot
        ixs = np.arange(0, data.shape[1])
        color = 'blue'
        fig, ax = plt.subplots(1, figsize=(90, 30))
        plt.plot(data.loc[["2%", "98%"], :].T, 'm--',
                label=["percentile 2", "percentile 98"], linewidth=1)
        plt.fill_between(ixs, data.loc["2%", :], data.loc["98%", :],
                        facecolor='gray', alpha=0.2)
        plt.plot(data.loc["mean", :].T, 'm--', label=f"{cell} means", linewidth=5)
        plt.plot(data.loc["current", :].T, 'r', label="current cell", linewidth=10)
        plt.ylabel('feature value', color=color)
        plt.xlabel('feature index', color=color)
        ax.set_yscale('log')
        ax.xaxis.set_major_locator(mticker.FixedLocator(ixs))
        ax.set_xticklabels(labels=list(ixs))
        ax.legend(loc=position)
        ax.grid(True)
        ax.grid(color='b', alpha=0.5, linestyle='dashed', linewidth=1)
        ax.set_title(f'Current cell features and mean features from {cell}s', color=color)
        ax.tick_params(axis='x', colors=color, rotation=0)
        ax.tick_params(axis='y', colors=color)

        return fig


    # Extraction of figures and dataframes for current cell features comparison
    def feat_fcn(cell):

        # Check cell type
        if cell == "monocyte":
            data = monocyte_dfs.append(current_features, ignore_index=True)
        elif cell == "lymphocyte":
            data = lymphocyte_dfs.append(current_features, ignore_index=True)
        elif cell == "hcl":
            data = hcl_dfs.append(current_features, ignore_index=True)
        elif cell == "apl":
            data = apl_dfs.append(current_features, ignore_index=True)

        data.index = ["count", "mean", "std", "min", "25%", "50%", "75%",
                    "max", "2%", "98%", "current"]

        # Arrange dataframe of shape features (1)
        df1, data1 = make_df(data, shape1)

        # Arrange dataframe of shape features (2)
        df2, data2 = make_df(data, shape2)

        # Arrange plot of shape features (1)
        fig1 = plot_arrange(data1, cell, 'upper left')

        # Arrange plot of shape features (2)
        fig2 = plot_arrange(data2, cell, 'upper left')

        # Arrange dataframe of first order features (1)
        df3, data3 = make_df(data, f_ord1)

        # Arrange plot of first order features (1)
        # data3.iloc[:, 2] = data3.iloc[:, 2] / 1e6  # Correction of MJ
        # data3.iloc[:, 19] = data3.iloc[:, 19] / 1e6  # Correction of MJ
        data3.iloc[:, 31] = data3.iloc[:, 31] + 1  # Correction of skewness plot
        fig3 = plot_arrange(data3, cell, 'lower left')

        return [fig1, df1.round(4),
                fig2, df2.round(4),
                fig3, df3.round(4)]


    ###############################################################################
    # Graphical User Interface with Gradio
    ###############################################################################

    with gr.Blocks() as demo:
        gr.Markdown("# Mononucleated cell features comparison")

        # Image and mask tab
        with gr.Tab("Create mask for image"):
            with gr.Row():
                image_button = gr.Button("Calculate features")
            with gr.Row():
                image_input = gr.Image(shape=(IM_SIZE, IM_SIZE))
                image_output = gr.Image()
            gr.Markdown(value=info_text)

        # General information and plots tab

        # Monocytes vs lymphocytes
        with gr.Tab("General plots"):
            with gr.Tab("Monocytes vs Lymphocytes"):
                c1 = 'monocyte'
                c2 = 'lymphocyte'
                with gr.Tab("Shape features"):
                    _ = gr.Plot(value=fig01,
                                label=f"Shape features (1) in {c1}-{c2} comparison")
                    _ = gr.Dataframe(value=df_s1_mon.round(4),
                                    label=f"Shape features (1) from {c1}s")
                    _ = gr.Dataframe(value=df_s1_lym.round(4),
                                    label=f"Shape features (1) from {c2}s")
                    _ = gr.Plot(value=fig02,
                                label=f"Shape features (2) in {c1}-{c2} comparison")
                    _ = gr.Dataframe(value=df_s2_mon.round(4),
                                    label=f"Shape features (2) from {c1}")
                    _ = gr.Dataframe(value=df_s2_lym.round(4),
                                    label=f"Shape features (2) from {c2}")
                with gr.Tab("First order pixel-value features"):
                    _ = gr.Plot(value=fig03,
                                label=f"Pixel-value features in {c1}-{c2} comparison")
                    _ = gr.Dataframe(value=df_pv_mon.round(4),
                                    label=f"Pixel-value features from {c1}")
                    _ = gr.Dataframe(value=df_pv_lym.round(4),
                                    label=f"Pixel-value features from {c2}")

            # Monocytes vs HCL
            with gr.Tab("Monocytes vs HCL"):
                c1 = 'monocyte'
                c2 = 'HCL'
                with gr.Tab("Shape features"):
                    _ = gr.Plot(value=fig04,
                                label=f"Shape features (1) in {c1}-{c2} comparison")
                    _ = gr.Dataframe(value=df_s1_mon.round(4),
                                    label=f"Shape features (1) from {c1}s")
                    _ = gr.Dataframe(value=df_s1_hcl.round(4),
                                    label=f"Shape features (1) from {c2}s")
                    _ = gr.Plot(value=fig05,
                                label=f"Shape features (2) in {c1}-{c2} comparison")
                    _ = gr.Dataframe(value=df_s2_mon.round(4),
                                    label=f"Shape features (2) from {c1}")
                    _ = gr.Dataframe(value=df_s2_hcl.round(4),
                                    label=f"Shape features (2) from {c2}")
                with gr.Tab("First order pixel-value features"):
                    _ = gr.Plot(value=fig06,
                                label=f"Pixel-value features in {c1}-{c2} comparison")
                    _ = gr.Dataframe(value=df_pv_mon.round(4),
                                    label=f"Pixel-value features from {c1}")
                    _ = gr.Dataframe(value=df_pv_hcl.round(4),
                                    label=f"Pixel-value features from {c2}")

            # Monocytes vs APL
            with gr.Tab("Monocytes vs APL"):
                c1 = 'monocyte'
                c2 = 'APL'
                with gr.Tab("Shape features"):
                    _ = gr.Plot(value=fig07,
                                label=f"Shape features (1) in {c1}-{c2} comparison")
                    _ = gr.Dataframe(value=df_s1_mon.round(4),
                                    label=f"Shape features (1) from {c1}s")
                    _ = gr.Dataframe(value=df_s1_apl.round(4),
                                    label=f"Shape features (1) from {c2}s")
                    _ = gr.Plot(value=fig08,
                                label=f"Shape features (2) in {c1}-{c2} comparison")
                    _ = gr.Dataframe(value=df_s2_mon.round(4),
                                    label=f"Shape features (2) from {c1}")
                    _ = gr.Dataframe(value=df_s2_apl.round(4),
                                    label=f"Shape features (2) from {c2}")
                with gr.Tab("First order pixel-value features"):
                    _ = gr.Plot(value=fig09,
                                label=f"Pixel-value features in {c1}-{c2} comparison")
                    _ = gr.Dataframe(value=df_pv_mon.round(4),
                                    label=f"Pixel-value features from {c1}")
                    _ = gr.Dataframe(value=df_pv_apl.round(4),
                                    label=f"Pixel-value features from {c2}")

            # Lymphocytes vs HCL
            with gr.Tab("Lymphocytes vs HCL"):
                c1 = 'lymphocyte'
                c2 = 'HCL'
                with gr.Tab("Shape features"):
                    _ = gr.Plot(value=fig10,
                                label=f"Shape features (1) in {c1}-{c2} comparison")
                    _ = gr.Dataframe(value=df_s1_lym.round(4),
                                    label=f"Shape features (1) from {c1}s")
                    _ = gr.Dataframe(value=df_s1_hcl.round(4),
                                    label=f"Shape features (1) from {c2}s")
                    _ = gr.Plot(value=fig11,
                                label=f"Shape features (2) in {c1}-{c2} comparison")
                    _ = gr.Dataframe(value=df_s2_lym.round(4),
                                    label=f"Shape features (2) from {c1}")
                    _ = gr.Dataframe(value=df_s2_hcl.round(4),
                                    label=f"Shape features (2) from {c2}")
                with gr.Tab("First order pixel-value features"):
                    _ = gr.Plot(value=fig12,
                                label=f"Pixel-value features in {c1}-{c2} comparison")
                    _ = gr.Dataframe(value=df_pv_lym.round(4),
                                    label=f"Pixel-value features from {c1}")
                    _ = gr.Dataframe(value=df_pv_hcl.round(4),
                                    label=f"Pixel-value features from {c2}")

            # Lymphocytes vs APL
            with gr.Tab("Lymphocytes vs APL"):
                c1 = 'lymphocyte'
                c2 = 'APL'
                with gr.Tab("Shape features"):
                    _ = gr.Plot(value=fig13,
                                label=f"Shape features (1) in {c1}-{c2} comparison")
                    _ = gr.Dataframe(value=df_s1_lym.round(4),
                                    label=f"Shape features (1) from {c1}s")
                    _ = gr.Dataframe(value=df_s1_apl.round(4),
                                    label=f"Shape features (1) from {c2}s")
                    _ = gr.Plot(value=fig14,
                                label=f"Shape features (2) in {c1}-{c2} comparison")
                    _ = gr.Dataframe(value=df_s2_lym.round(4),
                                    label=f"Shape features (2) from {c1}")
                    _ = gr.Dataframe(value=df_s2_apl.round(4),
                                    label=f"Shape features (2) from {c2}")
                with gr.Tab("First order pixel-value features"):
                    _ = gr.Plot(value=fig15,
                                label=f"Pixel-value features in {c1}-{c2} comparison")
                    _ = gr.Dataframe(value=df_pv_lym.round(4),
                                    label=f"Pixel-value features from {c1}")
                    _ = gr.Dataframe(value=df_pv_apl.round(4),
                                    label=f"Pixel-value features from {c2}")

            # HCL vs APL
            with gr.Tab("HCL vs APL"):
                c1 = 'HCL'
                c2 = 'APL'
                with gr.Tab("Shape features"):
                    _ = gr.Plot(value=fig16,
                                label=f"Shape features (1) in {c1}-{c2} comparison")
                    _ = gr.Dataframe(value=df_s1_hcl.round(4),
                                    label=f"Shape features (1) from {c1}s")
                    _ = gr.Dataframe(value=df_s1_apl.round(4),
                                    label=f"Shape features (1) from {c2}s")
                    _ = gr.Plot(value=fig17,
                                label=f"Shape features (2) in {c1}-{c2} comparison")
                    _ = gr.Dataframe(value=df_s2_hcl.round(4),
                                    label=f"Shape features (2) from {c1}")
                    _ = gr.Dataframe(value=df_s2_apl.round(4),
                                    label=f"Shape features (2) from {c2}")
                with gr.Tab("First order pixel-value features"):
                    _ = gr.Plot(value=fig18,
                                label=f"Pixel-value features in {c1}-{c2} comparison")
                    _ = gr.Dataframe(value=df_pv_hcl.round(4),
                                    label=f"Pixel-value features from {c1}")
                    _ = gr.Dataframe(value=df_pv_apl.round(4),
                                    label=f"Pixel-value features from {c2}")

        # Monocyte tab
        with gr.Tab("Monocyte comparison"):
            update_button01 = gr.Button("Update")
            mon_plot_1 = gr.Plot(label="Shape features (1)")
            mon_df_1 = gr.Dataframe(label="Shape features (1)")
            mon_plot_2 = gr.Plot(label="Shape features (2)")
            mon_df_2 = gr.Dataframe(label="Shape features (2)")
            mon_plot_3 = gr.Plot(label="Pixel-value features")
            mon_df_3 = gr.Dataframe(label="Pixel-value features")

        # Lymphocyte tab
        with gr.Tab("Lymphocyte comparison"):
            update_button02 = gr.Button("Update")
            lym_plot_1 = gr.Plot(label="Shape features (1)")
            lym_df_1 = gr.Dataframe(label="Shape features (1)")
            lym_plot_2 = gr.Plot(label="Shape features (2)")
            lym_df_2 = gr.Dataframe(label="Shape features (2)")
            lym_plot_3 = gr.Plot(label="Pixel-value features")
            lym_df_3 = gr.Dataframe(label="Pixel-value features")

        # HCL tab
        with gr.Tab("HCL comparison"):
            update_button03 = gr.Button("Update")
            hcl_plot_1 = gr.Plot(label="Shape features (1)")
            hcl_df_1 = gr.Dataframe(label="Shape features (1)")
            hcl_plot_2 = gr.Plot(label="Shape features (2)")
            hcl_df_2 = gr.Dataframe(label="Shape features (2)")
            hcl_plot_3 = gr.Plot(label="Pixel-value features")
            hcl_df_3 = gr.Dataframe(label="Pixel-value features")

        # APL tab
        with gr.Tab("APL comparison"):
            update_button04 = gr.Button("Update")
            apl_plot_1 = gr.Plot(label="Shape features (1)")
            apl_df_1 = gr.Dataframe(label="Shape features (1)")
            apl_plot_2 = gr.Plot(label="Shape features (2)")
            apl_df_2 = gr.Dataframe(label="Shape features (2)")
            apl_plot_3 = gr.Plot(label="Pixel-value features")
            apl_df_3 = gr.Dataframe(label="Pixel-value features")

        # Listeners
        image_input.change(create_mask, inputs=image_input, outputs=image_output)
        image_button.click(calc_features, inputs=None, outputs=[mon_plot_1,
                        mon_df_1, mon_plot_2, mon_df_2, mon_plot_3, mon_df_3,
                        lym_plot_1, lym_df_1, lym_plot_2, lym_df_2, lym_plot_3,
                        lym_df_3, hcl_plot_1, hcl_df_1, hcl_plot_2, hcl_df_2,
                        hcl_plot_3, hcl_df_3, apl_plot_1, apl_df_1, apl_plot_2,
                        apl_df_2, apl_plot_3, apl_df_3])
        update_button01.click(update1, inputs=None, outputs=[mon_plot_1, mon_df_1,
                            mon_plot_2, mon_df_2, mon_plot_3, mon_df_3])
        update_button02.click(update2, inputs=None, outputs=[lym_plot_1, lym_df_1,
                            lym_plot_2, lym_df_2, lym_plot_3, lym_df_3])
        update_button03.click(update3, inputs=None, outputs=[hcl_plot_1, hcl_df_1,
                            hcl_plot_2, hcl_df_2, hcl_plot_3, hcl_df_3])
        update_button04.click(update4, inputs=None, outputs=[apl_plot_1, apl_df_1,
                            apl_plot_2, apl_df_2, apl_plot_3, apl_df_3])


    demo.launch()
