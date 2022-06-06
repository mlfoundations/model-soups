'''
Datasets is based on https://github.com/mlfoundations/wise-ft/tree/master/src/datasets
'''
from .imagenet import *
from .imagenetv2 import ImageNetV2
from .imagenet_a import ImageNetAValClasses, ImageNetA
from .imagenet_r import ImageNetRValClasses, ImageNetR
from .objectnet import ObjectNetValClasses, ObjectNet
from .imagenet_sketch import ImageNetSketch