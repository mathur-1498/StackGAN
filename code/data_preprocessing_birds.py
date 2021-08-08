"""
Author: Vishal Kundar and Yash Mathur

data_preprocessing.py prepares the images for training the model.
Steps followed are:

1. Loading bounding boxes and creating relation between image files.
2. Cropping the image using the bounding boxes given with data set.
3. Loading the split train and test data from pickle file.
4. Storing high res and low res images after being cropped.

We use openCV to handle the images
"""
# Packages being used
import pickle
import pandas as pd
import numpy as np
import cv2

import PIL
from PIL import Image

# Defining some variables
lr_hr_retio = 4  # The ratio between low and high resolution images to be used for training
"""
imsize = 256
load_size = int(imsize * 76 / 64)
lr_size = int(load_size / lr_hr_retio)
"""
load_size = 256
lr_size = 64

def loadBbox():
    """
    Function to load bounding boxes and store in dictionary corresponding to image name

    PARAMETERS
    ----------
    NONE

    RETURNS
    -------
    bboxes: TYPE: Python dictionary
        DESCRIPTION: Dictionary consisting of boudning boxes of the corresponding image files.
    """
    # Loading boudning boxes from txt file
    bbox_path = '/Users/vishalkundar/Desktop/ML/ML_Projects/GAN_Project/Code/Data/bounding_boxes.txt'
    bbox_df = pd.read_csv(bbox_path, delim_whitespace=True,
                          header=None).astype(int)

    # Loading image file names from txt file
    imagetxt_path = '/Users/vishalkundar/Desktop/ML/ML_Projects/GAN_Project/Code/Data/images.txt'
    imFileName_df = pd.read_csv(
        imagetxt_path, delim_whitespace=True, header=None)
    imFilesList = imFileName_df[1].tolist()  # Converting to list

    # Creating a dictionary to store bbox for the corresponding image file
    bboxes = {x[:-4]: [] for x in imFilesList}  # Key set to image file name

    for i in range(len(imFilesList)):
        # Bounding box format in txt file -> [x-left, y-top, width, height]
        bbox = bbox_df.iloc[i][1:].tolist()

        # Setting value in dictionary
        key = imFilesList[i][:-4]
        bboxes[key] = bbox

    return bboxes


def load_filenames(training_data_path):
    """
    Since train and test split is done by the dataset author, we use the pickle
    file given to load the names of the image files corresponding to train/test
    set.

    PARAMETERS
    ----------
    training_data_path: TYPE: Python string
            DESCRIPTION: String containing path to pickle file of training/test
            set

    RETURNS
    -------
    filenames: TYPE: Python list
            DESCRIPTION: List of image file names corresponding to train or test 
            set.
    """
    # Path to pickle file
    filepath = training_data_path + 'filenames.pickle'

    # Opening file and storing content to list
    with open(filepath, 'rb') as f:
        filenames = pickle.load(f)

    return filenames


def get_image(image_file, image_size, bbox):
    """
    This function is used to read file, convert colorized images, crop the image
    based one bounding boxes and transform it into array

    PARAMETERS
    ----------
    image_file: TYPE: Python string
        DESCRIPTION: Name of image file

    image_size: TYPE: float
        DESCRIPTION: Size of the image file     

    bbox: TYPE: Python list
        DESCRIPTION: List containing info about bounding box of an image

    RETURNS
    -------
    image: TYPE: Numpy array
        DESCRIPTION: Image in numpy array format

    """
    img = cv2.imread(image_file)
    if len(img.shape) == 0:
        raise ValueError(image_file + " got loaded as a dimensionless array!")
           
    img = img.astype(np.float)

    # colorize image
    if img.ndim == 2:
        img = img.reshape(img.shape[0], img.shape[1], 1)
        img = np.concatenate([img, img, img], axis=2)
    if img.shape[2] == 4:
        img = img[:, :, 0:3]

    # Cropping image based on bounding box
    imsiz = img.shape
    # Getting center coordinates
    center_x = int((2 * bbox[0] + bbox[2]) / 2)
    center_y = int((2 * bbox[1] + bbox[3]) / 2)
    # Getting distance of box from center
    R = int(np.maximum(bbox[2], bbox[3]) * 0.75)
    # Setting x1, y1, x2, y2
    y1 = np.maximum(0, center_y - R)
    y2 = np.minimum(imsiz[0], center_y + R)
    x1 = np.maximum(0, center_x - R)
    x2 = np.minimum(imsiz[1], center_x + R)
    img_cropped = img[y1:y2, x1:x2, :]
    
    # Resizing image
    image = cv2.resize(img_cropped, (image_size, image_size), interpolation=cv2.INTER_LINEAR)

    return image

def hrlrGenerator(bboxes, train_filenames):
    """
    This function is used to get the original high resolution image and then convert it
    to low resolution and save both. This helps training the GAN better

    PARAMETERS
    ----------
    bboxes: TYPE: Python dictionary
        DESCRIPTION: Dictionary consisting of bounding boxes of the corresponding image files.

    train_filenames: TYPE: Python list
        DESCRIPTION: Contains list of image location    

    RETURNS
    -------
    hr_images: TYPE: Python list
            DESCRIPTION: List of high resolution images converted to array format

    lr_images: TYPE: Python list
            DESCRIPTION: List of low resolution images converted to array format
    """
    # List to store images
    hr_images = []
    lr_images = []

    # Loading images
    file_path = '/Users/vishalkundar/Desktop/ML/ML_Projects/GAN_Project/Data/birds/'
    for key in train_filenames:
        bbox = bboxes[key]
        filename = file_path + '/CUB_200_2011/CUB_200_2011/images/' + \
            key + '.jpg'  # Location to images

        # Sending image to crop according to bounding box and return array of pixels
        img = get_image(filename, load_size, bbox)
        img = img.astype('uint8')

        # Storing to list
        hr_images.append(img)
        #lr_img = cv2.resize(img, (lr_size, lr_size), interpolation=cv2.INTER_CUBIC)
        img = get_image(filename, lr_size, bbox)
        img = img.astype('uint8')
        lr_images.append(img)

    return hr_images, lr_images


def preprocessImages():
    """
    Central function to preprocess images. 

    PARAMETERS
    ----------
    NONE

    RETURNS
    -------
    NONE
    """
    # Loading the bounding boxes using txt file given with data
    bboxes = loadBbox()

    # Since the data given is already split into train and test set
    # We carry out their preprocessing individually

    # For training data
    train_data_path = '/Users/vishalkundar/Desktop/ML/ML_Projects/GAN_Project/Data/birds/train/'
    # Loading file names
    train_filenames = load_filenames(train_data_path)
    # Getting high res and low res images
    hr_images, lr_images = hrlrGenerator(bboxes, train_filenames)
    # Storing to pickle file
    outfile = train_data_path + str(load_size) + 'images.pickle'
    with open(outfile, 'wb') as f_out:
        pickle.dump(hr_images, f_out)

    outfile = train_data_path + str(lr_size) + 'images.pickle'
    with open(outfile, 'wb') as f_out:
        pickle.dump(lr_images, f_out)

    # For test data
    test_data_path = '/Users/vishalkundar/Desktop/ML/ML_Projects/GAN_Project/Data/birds/test/'
    test_filenames = load_filenames(test_data_path)
    hr_images, lr_images = hrlrGenerator(bboxes, test_filenames)

    outfile = test_data_path + str(load_size) + 'images.pickle'
    with open(outfile, 'wb') as f_out:
        pickle.dump(hr_images, f_out)

    outfile = test_data_path + str(lr_size) + 'images.pickle'
    with open(outfile, 'wb') as f_out:
        pickle.dump(lr_images, f_out)


if __name__ == '__main__':
    # Directory of dataset
    #BIRD_DIR = '/Users/vishalkundar/Desktop/ML/ML_Projects/GAN_Project/Data/birds'
    #FLOWER_DIR = '/Users/vishalkundar/Desktop/ML/ML_Projects/GAN_Project/Data/flowers'
    preprocessImages()
