#!/usr/bin/env python
# coding: utf-8

# # Top tagging model with ResNet-50 

# This is a training script about using top tagging data to train a ResNet-50 Neuron Network. <br>
# The training is base on Keras and using tensorflow backend to run.

# ## Import necessary package and do gpu test.


import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd 
import hdf5plugin, h5py, os, glob, datetime, tables, time, threading, sys
from keras.utils import np_utils
import tensorflow as tf
from tensorflow.keras.utils import plot_model
from IPython.display import Image
from sklearn.metrics import roc_auc_score
from keras.callbacks import Callback
import tensorflow_io as tfio 

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  # Restrict TensorFlow to only use the first GPU
    try:
        tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
        tf.config.experimental.set_virtual_device_configuration(
        gpus[0],
        [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=6000)])
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU")
    except RuntimeError as e:
    # Visible devices must be set before GPUs have been initialized
        print(e)

init = int(sys.argv[1])
fin = int(sys.argv[2])
phase = sys.argv[3]

# ## Construsting a sequential 
# Construst a model with Keras pre-configured ResNet-50 and one Dense layer with Softmax activation function. <br>
# Using Earlystopping and custom callback function to prevent training from overfitting and collect ROC/AUC data point. 


def  create_model():
    model = tf.keras.models.Sequential([
    tf.keras.applications.ResNet50(weights=None, pooling='max',classes=2048),
    tf.keras.layers.Dense(1024,activation='relu'),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(2, activation='softmax'),
    ])
    model.load_weights=('../aws-fpga-top-tagging-notebooks/dataset_Tom/weights-floatingpoint-224x224-fixval-best/class_weights.h5')
    model.compile(optimizer='adam', loss='binary_crossentropy',metrics=[tf.keras.metrics.AUC(num_thresholds=200, curve='ROC')])
    model.summary()
    model.save('./model/')
    model.save('./model/tmp_model.h5')
    return model



# ## Input data with H5py package
# Input data size: 1 channel image with shape 224x224. <br>
# Input label size: 1 or 0 with onehot encoding. <br>
# \# of input data(train(with val), test): 30000,6000


prefix = "/home/david/Storage/dataset_Tom/img224_all/converted/rotation_224_v1/"
train_filename = np.asanyarray(glob.glob(os.path.join(prefix,"train_*")))
test_filename = np.asanyarray(glob.glob(os.path.join(prefix,"test_*")))
val_filename = np.asanyarray(glob.glob(os.path.join(prefix,"val_*")))
tmp_prefix='/home/david/Storage/'
print(type(train_filename), len(train_filename))
print(type(test_filename), len(test_filename))
print(type(val_filename), len(val_filename))


def get_data(filename):
    with h5py.File(filename,'r') as f:
        img_, label_ = [], []
        pt_tmp = f['img_pt']
        label_tmp = f['label']
        for a in range(len(pt_tmp)):
            img_.append(pt_tmp[a])
        for b in range(len(label_tmp)):
            label_.append(label_tmp[b])  
    return np.asanyarray(img_), np.asanyarray(label_)

def write_data(FILE_NAME, im, lal, mode):
    out = tables.open_file(FILE_NAME, mode="w", title="Test file") 
    out.create_array("/", "image", im)
    out.create_array("/","label", lal)
    out.close()
    print("A temporary {0} file has been created.".format(mode))

def remove_data(FILE_NAME):
    if  os.path.exists(FILE_NAME) == True:
        os.remove(FILE_NAME)

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

def Preprocessing(filename, mode, Batch):
    print("+--------------------------------------------------------------------------------------------+")
    print("Starting data pre-processing")
    tStart = time.time()#計時開始
    file = ""
    seq = ("tmp_", mode, ".h5")
    
    im, lal = get_data(filename)
    #print(type(im), im.shape, len(im), type(lal), len(lal))
    im, lal = normalize_and_rgb(im, lal)
    print("Parsing data wtih length: {0}".format( len(im) ))
    
    r = threading.Thread( target=remove_data(os.path.join(tmp_prefix,file.join(seq))) )
    r.start()
    r.join
    
    os.path.join(os.path.join(prefix,file.join(seq)))
    
    t = threading.Thread( target = write_data(os.path.join(os.path.join(tmp_prefix,file.join(seq))), im, lal, mode) )
    t.start()
    t.join()
    tEnd = time.time()
    print("Pre-processing duration: {0}s".format(tEnd - tStart))
    return int(int(len(im))/Batch)
"""
def loop_Preprocessing(filename, loop, mode):
    print("+--------------------------------------------------------------------------------------------+")
    print("Starting data pre-processing")
    file = ""
    seq = ("tmp_", mode, ".h5")
    #if os.path.exists(os.path.join(prefix,file.join(seq))) == True : 
    #    os.remove(os.path.join(prefix,file.join(seq)))
    for i in range(loop):
        im, lal = get_data(filename[i])
        #print(type(im), im.shape, len(im), type(lal), len(lal))
        im, lal = normalize_and_rgb(im, lal)
        print(len(im),len(lal))
    #print(im ,lal)
        print("Parsing data wtih length: {0}".format( len(im) ))
        os.path.join(prefix,os.path.join(prefix,file.join(seq)))
        out = tables.open_file(os.path.join(prefix,os.path.join(prefix,file.join(seq))), mode="w", title="Test file")
        out.create_array("/", "image", im)
        out.create_array("/","label", lal)
        print("A temporary {0} file has been created.".format(mode))
    
    return int(int(len(im))/Batch)
"""
#def train_model(x,y):
#    logdir = os.path.join("logs","fit", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"),)
#    tensorboard_callback = tf.keras.callbacks.TensorBoard(logdir, histogram_freq=1, write_graph=True, write_images=True, )
#    train_history = model.fit(x, y, epochs=1, batch_size=64, shuffle=True, callbacks=[tensorboard_callback])
#    #model.summary()
#    return train_history, model




batch_zise = 32

#logdir = os.path.join("logs","fit", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"),)
#tensorboard_callback = tf.keras.callbacks.TensorBoard(logdir, histogram_freq=1, write_graph=False, write_images=False, )

#train stage      
def write_train_log(train_history):
    with open('train_log.txt', 'a') as f:
        auc = train_history.history['auc']
        loss = train_history.history['loss']
        f.writelines("{0:.3f}, {1:.3f}\n".format(auc[0], loss[0]))
def write_train_log_val(train_history):
    with open('train_log_val.txt', 'a') as g:
        val_auc = train_history.history['val_auc']
        val_loss = train_history.history['val_loss']
        g.writelines("{0:.3f}, {1:.3f}\n".format(val_auc[0], val_loss[0]))

def train_sub_0(round, batch_zise):
    model = create_model()
    plot_model(model, to_file='./Picture/Sequential_Model.png')
    Image('./Picture/Sequential_Model.png')
    STEP_train = Preprocessing(train_filename[round],"train", batch_zise)
    image_train = tfio.IODataset.from_hdf5(filename=os.path.join(tmp_prefix,'tmp_train.h5'), dataset="/image")
    label_train = tfio.IODataset.from_hdf5(filename=os.path.join(tmp_prefix,'tmp_train.h5'), dataset="/label")
    train = tf.data.Dataset.zip((image_train,label_train)).batch(batch_zise, drop_remainder=True).prefetch(tf.data.experimental.AUTOTUNE)
    print("+--------------------------------------------------------------------------------------------+")
    print("Round {0}".format(round+1))
    model = tf.keras.models.load_model('./model/tmp_model.h5')
    train_history = model.fit(train, epochs=1, steps_per_epoch=STEP_train,  shuffle=True)
    model.save('./model/tmp_model.h5')
    model.save_weights('model/tmp_weight.h5')
    write_log_sbu_1 = threading.Thread(target=write_train_log(train_history)) 
    write_log_sbu_1.start()
    write_log_sbu_1.join()


def train_sub_1(round, batch_zise):
    STEP_train = Preprocessing(train_filename[round],"train", batch_zise)
    image_train = tfio.IODataset.from_hdf5(filename=os.path.join(tmp_prefix,'tmp_train.h5'), dataset="/image")
    label_train = tfio.IODataset.from_hdf5(filename=os.path.join(tmp_prefix,'tmp_train.h5'), dataset="/label")
    train = tf.data.Dataset.zip((image_train,label_train)).batch(batch_zise, drop_remainder=True).prefetch(tf.data.experimental.AUTOTUNE)
    print("+--------------------------------------------------------------------------------------------+")
    print("Round {0}".format(round+1))
    model = tf.keras.models.load_model('./model/tmp_model.h5')
    model.load_weights("model/tmp_weight.h5")
    train_history = model.fit(train, epochs=1, steps_per_epoch=STEP_train,  shuffle=True)
    model.save('./model/tmp_model.h5')
    model.save_weights('model/tmp_weight.h5')
    write_log_sbu_1 = threading.Thread(target=write_train_log(train_history)) 
    write_log_sbu_1.start()
    write_log_sbu_1.join()

def train_sub_2(round, k, batch_zise):
    STEP_train = Preprocessing(train_filename[round],"train", batch_zise)
    STEP_val = Preprocessing(val_filename[k],"val", batch_zise)
    image_train = tfio.IODataset.from_hdf5(filename=os.path.join(tmp_prefix,'tmp_train.h5'), dataset="/image")
    label_train = tfio.IODataset.from_hdf5(filename=os.path.join(tmp_prefix,'tmp_train.h5'), dataset="/label")
    image_val = tfio.IODataset.from_hdf5(filename=os.path.join(tmp_prefix,'tmp_val.h5'), dataset="/image")
    label_val = tfio.IODataset.from_hdf5(filename=os.path.join(tmp_prefix,'tmp_val.h5'), dataset="/label")
    train = tf.data.Dataset.zip((image_train,label_train)).batch(batch_zise, drop_remainder=True).prefetch(tf.data.experimental.AUTOTUNE)
    val = tf.data.Dataset.zip((image_val,label_val)).batch(batch_zise, drop_remainder=True).prefetch(tf.data.experimental.AUTOTUNE)
    print("+--------------------------------------------------------------------------------------------+")
    print("Round {0}".format(round+1))
    model = tf.keras.models.load_model('./model/tmp_model.h5')
    model.load_weights("model/tmp_weight.h5")
    train_history = model.fit(train, validation_data=val, epochs=1, steps_per_epoch=STEP_train, validation_steps=STEP_val,  shuffle=True)
    model.save('./model/tmp_model.h5')
    model.save_weights('model/tmp_weight.h5')
    write_log_sbu_2 = threading.Thread(target=write_train_log(train_history))
    write_log_sbu_2.start()
    write_log_sbu_2.join()
    write_log_sbu_2_1 = threading.Thread(target=write_train_log_val(train_history)) 
    write_log_sbu_2_1.start()
    write_log_sbu_2_1.join()

    
def train_process(round, batch_zise):
    j=0
    if (round == 0):
        sub_0 = threading.Thread(target=train_sub_0(round, batch_zise))
        sub_0.start()
        sub_0.join
    elif ( (round+1) % 3 != 0 and round != 0 or ((round+1) % 3 == 0 and (round+1) <= 6)):
        sub_1 = threading.Thread(target=train_sub_1(round, batch_zise))
        sub_1.start()
        sub_1.join
    elif ( (round+1) % 3 == 0 and round != 0 and (round+1) > 6 and round !=0 ):
        sub_2 = threading.Thread(train_sub_2(round, j, batch_zise))
        sub_2.start()
        sub_2.join()            
        j +=1





def write_test_log(score, mode):
    with open('{0}_log.txt'.format(mode), 'a') as f:
        sco = score[0]
        f.write("{0:.3f}\n".format(sco))
        
def test_evaluate(round, batch_zise):
    STEP_test = Preprocessing(test_filename[round],"test", batch_zise)
    image_test = tfio.IODataset.from_hdf5(filename=os.path.join(tmp_prefix,'tmp_test.h5'), dataset="/image")
    label_test = tfio.IODataset.from_hdf5(filename=os.path.join(tmp_prefix,'tmp_test.h5'), dataset="/label")
    test = tf.data.Dataset.zip((image_test,label_test)).batch(batch_zise, drop_remainder=True).prefetch(tf.data.experimental.AUTOTUNE)
    print("+--------------------------------------------------------------------------------------------+")
    print("Round {0}".format(round+1))
    model = tf.keras.models.load_model('./model/tmp_model.h5')
    model.load_weights("model/tmp_weight.h5")
    score = model.evaluate(test, steps=STEP_test)
    print(score[0])
    write_log_sbu_1 = threading.Thread(target=write_test_log(score, "test")) 
    write_log_sbu_1.start()
    write_log_sbu_1.join()

def test(round, batch_zise):
    test_ = threading.Thread( target=test_evaluate(round, batch_zise))
    test_.start()
    test_.join()


def main_process(init, fin, phase):
    if ( phase == "train"):
        for i in range(init, fin):
            train_threading = threading.Thread(target = train_process(i, batch_zise))
            train_threading.start()
            train_threading.join
    elif ( phase == "test"):
        for i in range(init, fin):
            test_stage = threading.Thread( target=test(i, batch_zise) )
            test_stage.start()
            test_stage.join()
    else :
        print("Please input a phase for run.")
    
main_process(init, fin, phase)
