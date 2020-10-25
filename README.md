# Spot the Mask challenge

<div align="center">
    <img src = "https://pbs.twimg.com/media/EVyYt-IU8AALA3x.jpg" 
     height = "500"
     width = "1000">
</div>

## Description

Details about the dataset and challenge can be seen on [Challenge page](https://zindi.africa/hackathons/spot-the-mask-challenge)


## Dataset

The data have been split into a test and training set. The training set contains ~1300 images and the test set contains 509 images. There are two types of images in this dataset, people or images with face masks and people or images without.

Your task is to provide the probability that an image contains at least one mask. For each unique image ID you should estimate the likelihood that the image contains at least one mask, with an estimated probability value between 0 and 1.

The dataset comprises of the following files:

- [images.zip (~193mb)](https://api.zindi.africa/v1/competitions/spot-the-mask-challenge/files/images.zip)
- [train_labels.csv](https://api.zindi.africa/v1/competitions/spot-the-mask-challenge/files/train_labels.csv)
- [sample_submission.csv](https://api.zindi.africa/v1/competitions/spot-the-mask-challenge/files/sample_sub_v2.csv)


## Pre-processing
Images in the dataset did not have fixed size therefore it was mandatory to resize them for training. Therefore after careful consideration and looking at memory constraints we decided to resize all the images to 224x224. Along with this Images were normalized before performing data augmentation.

The size of dataset is small so we needed to add more data for training and we used data augmentations. For Data Augmentation we performed:

- Rotation
- Zoom
- Shear 
- Width Shift
- Heights Shift

## Model Diagram
I used Transfer Learning and the NasNet-Mobile.
More information about [NasNet](https://sh-tsang.medium.com/review-nasnet-neural-architecture-search-network-image-classification-23139ea0425d)

## Installation
To get this repo work please install all the dependencies using the command below:
```
pip install -r requirments.txt
```