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
                    information['file'] += [annotation.split('/')[-1][0:-4]] 
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