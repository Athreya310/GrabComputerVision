import numpy as np
import os
import scipy.io
import cv2 as cv
import random

# USER INPUT REQUIRED - Set x as the working directory to the folder where the folder downloaded from github is saved.
x = r'C:\Users\ASUS\Documents'

# function that creates a folder if it does not exist
def create_folder(folder):
    if os.path.exists(folder) == False:
        os.makedirs(folder)
        
# Part 1: Transforming the data from the raw files
def extract_train_data():
    cars = scipy.io.loadmat('devkit/cars_train_annos')
    annotations = cars['annotations']
    annotations = np.transpose(annotations)
    imgs = []
    bboxes = []
    labels = []
    
    for i in annotations:
        # extract the dimension of the bounding boxes of the image
        bbox_x1 = i[0][0][0][0]
        bbox_y1 = i[0][1][0][0]
        bbox_x2 = i[0][2][0][0]
        bbox_y2 = i[0][3][0][0]
        # extract the label of the image
        label = i[0][4][0][0]
        # saves the label as standardised four digit numbers
        labels.append('%04d' % (label,))
        # extract name of the image
        img = i[0][5][0]
        bboxes.append((bbox_x1, bbox_y1, bbox_x2, bbox_y2))
        imgs.append(img)
    transform_train_data(imgs, labels, bboxes)

def extract_test_data():
    cars = scipy.io.loadmat('devkit/cars_test_annos')
    annotations = cars['annotations']
    annotations = np.transpose(annotations)
    imgs = []
    bboxes = []
    
    for i in annotations:
        # extract the dimension of the bounding boxes of the image
        bbox_x1 = i[0][0][0][0]
        bbox_y1 = i[0][1][0][0]
        bbox_x2 = i[0][2][0][0]
        bbox_y2 = i[0][3][0][0]
        # extract name of the image
        img = i[0][4][0]
        bboxes.append((bbox_x1, bbox_y1, bbox_x2, bbox_y2))
        imgs.append(img)
    transform_test_data(imgs, bboxes)

# Part 2: Saving the data into folders sorted by class
def transform_train_data(imgs, labels, bboxes):
    # This is where the raw images are saved
    raw_folder = 'cars_train'
    # Randomly select 80% of the images into the training set
    num_samples = len(imgs)
    train_split = 0.8
    num_train = int(round(num_samples * train_split))
    train_set = random.sample(range(num_samples), num_train)

    for i in range(num_samples):
        img = imgs[i]
        label = labels[i]
        (x1, y1, x2, y2) = bboxes[i]
        raw_path = os.path.join(raw_folder, img)
        raw_image = cv.imread(raw_path)
        height, width = raw_image.shape[:2]
        # 16 pixel margin to capture border features of cars
        margin = 16
        x1 = max(0, x1 - margin)
        y1 = max(0, y1 - margin)
        x2 = min(x2 + margin, width)
        y2 = min(y2 + margin, height)
        # Sort the image according to if it was selected to be a training or validation image.
        if i in train_set:
            sort_folder = 'data/training_set'
        else:
            sort_folder = 'data/validation_set'
        # Create folder to save the images to be trained on.
        sort_path = os.path.join(sort_folder, label)
        if os.path.exists(sort_path) == False:
            os.makedirs(sort_path)
        sort_path = os.path.join(sort_path, img)
        # Crop images and write to folders to model
        crop_image = raw_image[y1:y2, x1:x2]
        sort_img = cv.resize(src=crop_image, dsize=(img_height, img_width))
        cv.imwrite(sort_path, sort_img)

def transform_test_data(imgs, bboxes):
    raw_folder = 'cars_test'
    sort_folder = 'data/test_set'
    num_samples = len(imgs)

    for i in range(num_samples):
        img = imgs[i]
        (x1, y1, x2, y2) = bboxes[i]
        raw_path = os.path.join(raw_folder, img)
        raw_image = cv.imread(raw_path)
        height, width = raw_image.shape[:2]
        sort_path = os.path.join(sort_folder, img)
        crop_image = raw_image[y1:y2, x1:x2]
        sort_img = cv.resize(src=crop_image, dsize=(img_height, img_width))
        cv.imwrite(sort_path, sort_img)

# Change directory to the data folder
os.chdir(x + r'\GrabComputerVision\Raw Data')

if __name__ == '__main__':

    create_folder('data/training_set')
    create_folder('data/validation_set')
    create_folder('data/test_set')
    
    # parameters of the images
    img_width, img_height = 224, 224
    extract_train_data()
    extract_test_data()