# Mononucleated cells feature extractor

In this repository you will find a folder scheme with various scripts.

The following is a detailed description of the folder scheme and the content of each of the scripts:

The folder scheme is as follows:

* data: Folder with the dataset. You should have 2 subfolders named 'img' and 'msk', containing the images and masks respectively. In turn, they can contain as many subfolders as you want. The important thing is that the paired image and mask files should have the same name (except for the extension)

* output: Folder in which some results are automatically saved.

* pyrad_data: Folder where .csv files are stored. These files are generated in one of the scripts and read in another, so it is essential to have this folder for correct operation.

* src: Folder with the source code.

* LICENSE: File with the type of license.

* README: This text.

* unetpp_2ldice.pth: Trained model. It is a U-Net++ model that is based on the Dice coefficient of 2 of the 3 channels of the mask.

# ¿What to do?

This document reflects what can be done with the scripts in this repository. Basically, the work carried out as the final thesis of the Master in Bioinformatics and Biostatistics can be fully reproduced.

All scripts are extensively commented so that the code can be followed easily.

## Selection of the model

If you start from scratch, possibly the first thing you want to do is select the model that best suits the type of images. To do this, the script called "KFold_pipeline.py" must be executed in the "src" folder. This script implements a pipeline with K-Fold validation for 9 considered models, all except U2Net can be loaded from the _segmentation_models_pytorch_ library:

* U-Net [1]
* Pyramid Attention Network (PAN) [2]
* Feature Pyramid Network (FPN) [3]
* Pyramid Scene Parsing Network (PSPNet) [4]
* DeepLabV3+ [5]
* Multi-scale Attention Network (Ma-Net) [6]
* U-Net++ [7]
* U²Net [8]
* LinkNet [9]

The U²Net model can be downloaded from the link that appears in reference [8] (https://github.com/xuebinqin/U-2-Net); however, this will not be necessary because a copy has been left inside the "src/u2net" folder.

In the script you can configure some parameters, such as the K value of the KFold, the number of epochs to train, the size to which the images will be cropped, the number of mask classes, etc. It will be enough to follow the indications of the code comments.

A log of this process will be saved in the "output" folder with the name "comp.txt". The value of the average Dice coefficient of the K training sessions of each model will be recorded in this file.

If you have a large dataset and you want the process to be faster, you can use the dataset published by CellaVision blog that you will find in the link (https://github.com/zxaoyou/segmentation_WBC). Ref [10]. Use the one with 100 images; it's enough.

## Extraction of graphic features from the dataset

This step can be interchanged with the previous one. It is about the extraction of graphic features of the dataset, from the original images and their masks. To do this you have to run the script called "src/feature_extractor.py". The only important parameter is the directory in which the images and masks are located.

This script will generate data files in .csv format which will be saved in the "pyrad_data" folder.

## Training of the model

At this point, since the model for segmentation has already been selected and the features of the different groups have been extracted, we are ready to fine tune the model. To do this you have to run the script called "src/Full_pipeline.py".

This is a large script in which there are many parameters that can be configured. Even so, it is neatly commented and partitioned to make it easy to understand.

The result obtained, in addition to a few graphics, will be a file called "model.pth" with the trained model. When this file is completive, it is recommended to rename it to "unetpp_2ldice.pth" because this is the name used in the next step (next script), or maybe you'd prefer to modify the call to this model in the application script ("src/app.py").

## The app

The app (named "src/app.py") is the end result of this work. This application is a script that implements a graphical environment developed in Gradio. When launching it, open the browser and place the following in the address bar: http://localhost:7860/

In a few moments a screen will appear that allows you to drag and drop an image. By doing this, the corresponding mask will automatically be generated. Then, all you have to do is click on the "Calculate features" button and all the comparisons of the current cell with each of the cells considered (currently monocytes, lymphocytes, HCL cells and APL cells) will be obtained.

HCL: Hairy cell leukemia
APL: Acute promyelocyte leukemia


# References

[1] Ronneberger, O., Fischer, P., & Brox, T. (2015). U-net: Convolutional networks for biomedical image segmentation. Lecture Notes in Computer Science (Including Subseries Lecture Notes in Artificial Intelligence and Lecture Notes in Bioinformatics), 9351. https://doi.org/10.1007/978-3-319-24574-4_28

[2] Li, H., Xiong, P., An, J., & Wang, L. (2019). Pyramid attention network for semantic segmentation. British Machine Vision Conference 2018, BMVC 2018.

[3] Kirillov, A., Girshick, R., He, K., & Dollar, P. (2019). Panoptic feature pyramid networks. Proceedings of the IEEE Computer Society Conference on Computer Vision and Pattern Recognition, 2019-June. https://doi.org/10.1109/CVPR.2019.00656

[4] Zhao, H., Shi, J., Qi, X., Wang, X., & Jia, J. (2017). Pyramid scene parsing network. Proceedings - 30th IEEE Conference on Computer Vision and Pattern Recognition, CVPR 2017, 2017-January. https://doi.org/10.1109/CVPR.2017.660

[5] Chen, L. C., Zhu, Y., Papandreou, G., Schroff, F., & Adam, H. (2018). Encoder-decoder with atrous separable convolution for semantic image segmentation. Lecture Notes in Computer Science (Including Subseries Lecture Notes in Artificial Intelligence and Lecture Notes in Bioinformatics), 11211 LNCS. https://doi.org/10.1007/978-3-030-01234-2_49

[6] Fan, T., Wang, G., Li, Y., & Wang, H. (2020). Ma-net: A multi-scale attention network for liver and tumor segmentation. IEEE Access, 8. https://doi.org/10.1109/ACCESS.2020.3025372

[7] Zhou, Z., Rahman Siddiquee, M. M., Tajbakhsh, N., & Liang, J. (2018). Unet++: A nested u-net architecture for medical image segmentation. Lecture Notes in Computer Science (Including Subseries Lecture Notes in Artificial Intelligence and Lecture Notes in Bioinformatics), 11045 LNCS. https://doi.org/10.1007/978-3-030-00889-5_1

[8] Qin, X., Zhang, Z., Huang, C., Dehghan, M., Zaiane, O. R., & Jagersand, M. (2020). U2-Net: Going deeper with nested U-structure for salient object detection. Pattern Recognition, 106. https://doi.org/10.1016/j.patcog.2020.107404

[9] Chaurasia, A., & Culurciello, E. (2018). LinkNet: Exploiting encoder representations for efficient semantic segmentation. 2017 IEEE Visual Communications and Image Processing, VCIP 2017, 2018-January. https://doi.org/10.1109/VCIP.2017.8305148

[10] Zheng, X., Wang, Y., Wang, G., & Liu, J. (2018). Fast and robust segmentation of white blood cell images by self-supervised learning. Micron, 107. https://doi.org/10.1016/j.micron.2018.01.010
