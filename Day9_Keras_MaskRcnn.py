from mrcnn.config import Config
from mrcnn import model as modellib
from mrcnn import visualize
import numpy as np
import colorsys
import argparse
import imutils
import random
import cv2
import os

# first of all the required package for maskrcnn which is mrcnn
# mrcnn has been installed using official repo of the mask rcnn as following
# $ git clone https://github.com/matterport/Mask_RCNN.git
# $ cd Mask_RCNN
# $ python setup.py install

# credit to Pyimagesearch

print("hello there I am going to perform object detection using maskRcnn ")