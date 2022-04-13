# Data Manipulation
import pandas as pd

# Numerical Analysis
import numpy as np

# Data Visualization
from matplotlib import pyplot as plt
import seaborn as sns

# Operating System
import os

# Deep Learning and Object Detection
import tensorflow as tf
from tensorflow import keras
import cv2

# Data Extraction
import glob
from xml.etree import ElementTree

# define dics for testing and training

annotations_directory = '../data/annotations'
images_directory = '../data/images'

# check if path is correct
# check if folder does exist


from os.path import exists


# reading the data in with the help of https://www.kaggle.com/code/stpeteishii/face-mask-get-annotation-info-from-xml/notebook
isdir = os.path.isdir('data/annotations')
file_exist = exists('data/images/maksssksksss0.png')
print(isdir)
print(file_exist)

annotations_directory = 'data/annotations'
images_directory = '..data/images'

information = {'xmin': [], 'ymin': [], 'xmax': [], 'ymax': [], 'label': [], 'file': [], 'width': [], 'height': []}

for annotation in glob.glob(annotations_directory + '/*.xml'):
    tree = ElementTree.parse(annotation)
    
    for element in tree.iter():
        if 'size' in element.tag:
            for attribute in list(element):
                if 'width' in attribute.tag: 
                    width = int(round(float(attribute.text)))
                if 'height' in attribute.tag:
                    height = int(round(float(attribute.text)))    

        if 'object' in element.tag:
            for attribute in list(element):
                
                if 'name' in attribute.tag:
                    name = attribute.text                 
                    information['label'] += [name]
                    information['width'] += [width]
                    information['height'] += [height] 
                    a = annotation.replace("data/annotations\\", "")
                    b = a.replace(".xml","")

                    information['file'] += [b]
                    # information['file'] += [annotation.replace('data/annotations',"")] 
                if 'bndbox' in attribute.tag:
                    for dimension in list(attribute):
                        if 'xmin' in dimension.tag:
                            xmin = int(round(float(dimension.text)))
                            information['xmin'] += [xmin]
                        if 'ymin' in dimension.tag:
                            ymin = int(round(float(dimension.text)))
                            information['ymin'] += [ymin]                                
                        if 'xmax' in dimension.tag:
                            xmax = int(round(float(dimension.text)))
                            information['xmax'] += [xmax]                                
                        if 'ymax' in dimension.tag:
                            ymax = int(round(float(dimension.text)))
                            information['ymax'] += [ymax]

annotations_info_df = pd.DataFrame(information)
annotations_info_df.head(10)

# ======================= saved all the data for annotation
annotations_info_df['annotation_file'] = annotations_info_df['file'] + '.xml'
annotations_info_df['image_file'] = annotations_info_df['file'] + '.png'

# Tidy Grammatical Issue
annotations_info_df.loc[annotations_info_df['label'] == 'mask_weared_incorrect', 'label'] = 'mask_incorrectly_worn'
annotations_info_df


# ======================= cleaning and fixing data ^^^

# check if the label is right

# Function to Show Actual Image
def render_image(image):
    plt.figure(figsize = (12, 8))
    plt.imshow(image)
    plt.show()
    
# Function to Convert BGR to RGB
def convert_to_RGB(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

annotations_info_df['image_file'].iloc[0]

# check if 0 exist
image_0_path= 'data/images/' + annotations_info_df['image_file'].iloc[0]
image_0_path

file737_exist = exists('data/images/maksssksksss0.png')
print(image_0_path)

image_0 = cv2.imread(image_0_path)
image_0

render_image(convert_to_RGB(image_0))

annotation_737_path = 'data/annotations/' + annotations_info_df['annotation_file'].iloc[0]
annotation_737_path

image_0.shape

