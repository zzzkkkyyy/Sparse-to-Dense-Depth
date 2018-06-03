__author__ = 'charlie'
import os
import random
from tensorflow.python.platform import gfile
import glob
import numpy as np
from scipy.misc import imread, imresize
import tensorflow as tf

def read_dataset(data_dir):
    result = {}
    #filepath = os.path.join(data_dir, "NYU_Dataset")
    filepath = os.path.join(data_dir, "ADEChallengeData2016")
    if not os.path.exists(filepath):
        print("not found data file!")
        return None
    result = create_image_lists(filepath)
    return result['training'], result['validation']

def create_sparse_depth(depth, prob):
    mask_keep = np.random.uniform(0, 1, depth.shape) < prob
    sparse_depth = np.zeros(depth.shape)
    sparse_depth[mask_keep] = depth[mask_keep]
    return sparse_depth

def create_rgbd(rgb, depth, prob):
    sparse_depth = create_sparse_depth(depth, prob)
    rgbd = np.append(rgb, np.expand_dims(sparse_depth, axis = 2), axis = 2)
    return rgbd

def create_image_lists(image_dir):
    if not gfile.Exists(image_dir):
        print("Image directory '" + image_dir + "' not found.")
        return None
    
    directories = ['training', 'validation']
    threshold = {'training': 2000, 'validation': 50}
    image_list = {}
    for directory in directories:
        image_list[directory] = []
    file_list = []
    
    file_glob = os.path.join(image_dir, "images", "training", '*.' + 'jpg')
    file_list.extend(glob.glob(file_glob))
    if not file_list:
        print('No files found')
    
    directory = 'training'
    count = 0
    resize_width, resize_height = 128, 192
    for f in file_list:
        #print(f)
        image_f = imread(f)
        image_f = imresize(image_f, [resize_width, resize_height], interp = 'nearest')
        if count == threshold['training']:
            directory = 'validation'
        elif count == threshold['training'] + threshold['validation']:
            break
        filename = os.path.splitext(f.split("/")[-1])[0]
        anno = os.path.join(image_dir, "annotations", "training", filename + '.png')
        if os.path.exists(anno):
            image_anno = imread(anno)
            image_anno = imresize(image_anno, [resize_width, resize_height], interp = 'nearest')
            record = {'image': create_rgbd(rgb = image_f, depth = image_anno, prob = 0.1), 'annotation': image_anno}
            image_list[directory].append(record)
            count = count + 1
        else:
            print("No according annotation found for {}".format(filename))
    del file_list
    if (len(image_list) != 2):
        print('validation image not add.')
        return None
    else:
        for d in directories:
            random.shuffle(image_list[d])
            no_of_images = len(image_list[d])
            print('No. of {} files: {}'.format(d, no_of_images))
        return image_list
"""
def create_image_lists(image_dir):
    if not gfile.Exists(image_dir):
        print("Image directory '" + image_dir + "' not found.")
        return None
    
    directories = ['training', 'validation']
    threshold = {'training': 600, 'validation': 24}
    image_list = {}
    for directory in directories:
        image_list[directory] = []
    file_list = []
    annotation_list = []
    
    file_glob = os.path.join(image_dir, 'rgb', '*.' + 'ppm')
    annotation_glob = os.path.join(image_dir, 'depth', '*.' + 'pgm')
    file_list.extend(glob.glob(file_glob))
    annotation_list.extend(glob.glob(annotation_glob))
    if not file_list or not annotation_list:
        print('No files found')
        return image_list
    
    directory = 'training'
    count = 0
    file = zip(file_list, annotation_list)
    for f, anno in file:
        image_f = imread(f)
        image_anno = imread(anno)
        if count == threshold['training']:
            directory = 'validation'
        elif count == threshold['training'] + threshold['validation']:
            break
        record = {'image': create_rgbd(rgb = image_f, depth = image_anno, prob = 5e-3), 'annotation': image_anno}
        #record = {'image': f, 'annotation': anno}
        image_list[directory].append(record)
        count = count + 1
    del file_list, annotation_list, file
    if (len(image_list) != 2):
        print('validation image not add.')
        return None
    else:
        for d in directories:
            random.shuffle(image_list[d])
            no_of_images = len(image_list[d])
            print('No. of {} files: {}'.format(d, no_of_images))
        return image_list
"""
    