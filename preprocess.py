#!/usr/bin/env python3
import tables
import h5py 
import numpy as np
import matplotlib.pyplot as plt 
import hdf5plugin
import os
import glob
import tqdm

def normalize_and_rgb(images, label):
    #normalize image to 0-255 per image.
    image_sum = 1/np.sum(np.sum(images,axis=1),axis=-1)
    given_axis = 0
    # Create an array which would be used to reshape 1D array, b to have
    # singleton dimensions except for the given axis where we would put -1
    # signifying to use the entire length of elements along that axis
    dim_array = np.ones((1,images.ndim),int).ravel()
    dim_array[given_axis] = -1
    # Reshape b with dim_array and perform elementwise multiplication with
    # broadcasting along the singleton dimensions for the final output
    image_sum_reshaped = image_sum.reshape(dim_array)
    images = images*image_sum_reshaped*255

    # make it rgb by duplicating 3 channels.
    images = np.stack([images, images, images],axis=-1)

    return images, label

DATA_PREFIX = '/home/david/workplace/Data_from_Tom/img224_all/converted/rotation_224_v1/'
TRAIN_FILE_LIST = np.sort(np.array(glob.glob(os.path.join(DATA_PREFIX,"train_*.h5"))))
TEST_FILE_LIST = np.sort(np.array(glob.glob(os.path.join(DATA_PREFIX,"test_*.h5"))))
VAL_FILE_LIST = np.sort(np.array(glob.glob(os.path.join(DATA_PREFIX,"val_*.h5"))))

TRAIN_BATCH = len(TRAIN_FILE_LIST)
TEST_BATCH = len(TEST_FILE_LIST)
VAL_BATCH = len(VAL_FILE_LIST)

# for i in tqdm.trange(TRAIN_BATCH):
for i in range(1):
    with h5py.File(f'/home/david/workplace/Data_from_Tom/img224_all/converted/rotation_224_v1/train_file_{i}.h5', 'r') as file: 
        img = file['img_pt'][:]
        label = file['label'][:]
    nor_img, nor_label = normalize_and_rgb(img, label)
    if i == 0:
        with h5py.File("train_file_1_new.h5", 'a') as output:
            output.create_dataset("img", data=nor_img, maxshape=(None, 224, 224, 3), compression="gzip", compression_opts=9)
            output.create_dataset("label", data=nor_label, maxshape=(None, 2), compression="gzip", compression_opts=9)
    else: 
        with h5py.File("train_file_merge.h5", 'a') as output:
            output["img"].resize((output["img"].shape[0] + nor_img.shape[0]), axis = 0)
            output["label"].resize((output["label"].shape[0] + nor_label.shape[0]), axis = 0)
            output["img"][-nor_img.shape[0]:] = nor_img
            output["label"][-nor_label.shape[0]:] = nor_label
            
# for i in tqdm.trange(TEST_BATCH):
for i in range(1):
    with h5py.File(f'/home/david/workplace/Data_from_Tom/img224_all/converted/rotation_224_v1/test_file_{i}.h5', 'r') as file: 
        img = file['img_pt'][:]
        label = file['label'][:]
    nor_img, nor_label = normalize_and_rgb(img, label)
    if i == 0:
        with h5py.File("test_file_1_new.h5", 'a') as output:
            output.create_dataset("img", data=nor_img, maxshape=(None, 224, 224, 3), compression="gzip", compression_opts=9)
            output.create_dataset("label", data=nor_label, maxshape=(None, 2), compression="gzip", compression_opts=9)
    else: 
        with h5py.File("test_file_merge.h5", 'a') as output:
            output["img"].resize((output["img"].shape[0] + nor_img.shape[0]), axis = 0)
            output["label"].resize((output["label"].shape[0] + nor_label.shape[0]), axis = 0)
            output["img"][-nor_img.shape[0]:] = nor_img
            output["label"][-nor_label.shape[0]:] = nor_label
            
# for i in tqdm.trange(VAL_BATCH):
for i in range(1):
    with h5py.File(f'/home/david/workplace/Data_from_Tom/img224_all/converted/rotation_224_v1/val_file_{i}.h5', 'r') as file: 
        img = file['img_pt'][:]
        label = file['label'][:]
    nor_img, nor_label = normalize_and_rgb(img, label)
    if i == 0:
        with h5py.File("val_file_1_new.h5", 'a') as output:
            output.create_dataset("img", data=nor_img, maxshape=(None, 224, 224, 3), compression="gzip", compression_opts=9)
            output.create_dataset("label", data=nor_label, maxshape=(None, 2), compression="gzip", compression_opts=9)
    else: 
        with h5py.File("val_file_merge.h5", 'a') as output:
            output["img"].resize((output["img"].shape[0] + nor_img.shape[0]), axis = 0)
            output["label"].resize((output["label"].shape[0] + nor_label.shape[0]), axis = 0)
            output["img"][-nor_img.shape[0]:] = nor_img
            output["label"][-nor_label.shape[0]:] = nor_label