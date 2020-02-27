import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd 
import hdf5plugin 
import h5py 
import os
from keras.utils import np_utils
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator

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

train_img_pt = []
test_img_pt = []
train_label_ = []
test_label_ = []

prefix = "/home/david/ResNet-50-Keras/data/"
train_file = "train/train_file_"
val_file = "val/val_file_"
file_type = ".h5"
path = []
filename = ''
for i in range(5,10):
    seq =  (train_file, str(i),file_type)
    path = os.path.join(prefix, filename.join(seq))
    with h5py.File(path,'r') as f:
        pt_ = f['img_pt']
        label_ = f['label']
        for a in range(len(pt_)):
            train_img_pt.append(pt_[a])
        for b in range(len(label_)):
            train_label_.append(label_[b])
            
for i in range(0,1):
    seq =  (val_file, str(i),file_type)
    path = os.path.join(prefix, filename.join(seq))
    with h5py.File(path,'r') as f :
        pt_ = f['img_pt']
        label_ = f['label']
        for a in range(0,4000):
            test_img_pt.append(pt_[a])
        for b in range(0,4000):
            test_label_.append(label_[b])


def normalize_and_rgb(img, label):  
    output = np.stack([np.asanyarray(img)/ np.asanyarray(img).max(), np.asanyarray(img)/ np.asanyarray(img).max(), np.asanyarray(img)/ np.asanyarray(img).max()],axis=-1)
    yield output, np.asanyarray(label)
train_img = normalize_and_rgb(train_img_pt, train_label_)
test_img = normalize_and_rgb(test_img_pt, test_label_)

print(train_img, test_img)
#train_label = np.asanyarray(train_label_)
#test_label = np.asanyarray(test_label_)
#print(train_label.shape, test_label.shape)

from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.applications.resnet50 import ResNet50
from keras.callbacks import EarlyStopping,ModelCheckpoint
import keras

model = Sequential()
model.load_weights=('../aws-fpga-top-tagging-notebooks/dataset_Tom/weights-floatingpoint-224x224-fixval-best/class_weights.h5')
model.add(ResNet50(pooling='max'))
model.add(Dense(2, activation='softmax'))


model.summary()
modelcheckpoint = ModelCheckpoint('./model/ResNet-50-best.model',
                                                     monitor='val_acc', save_best_only=True)
earlystop = EarlyStopping()


from keras import losses
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
train_history = model.fit_generator(train_img, validation_data=test_img,\
                                    validation_freq=1, epochs=10, verbose=1,use_multiprocessing=True, steps_per_epoch=20000, validation_steps=4000)

#Define a function to show training history
def show_train_history(train_history, train, validation):
    plt.plot(train_history.history[train])
    plt.plot(train_history.history[validation])
    plt.title('Train history')
    plt.xlabel('Epoch')
    plt.ylabel(train)
    plt.legend(['train','validaiton'],loc='upper left')
    plt.show()

#Show the result of training 
show_train_history(train_history,'accuracy','val_accuracy')
show_train_history(train_history,'loss','val_loss')



