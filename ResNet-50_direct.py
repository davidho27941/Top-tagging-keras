#!/usr/bin/env python
# coding: utf-8

# # Top tagging model with ResNet-50 

# This is a training script about using top tagging data to train a ResNet-50 Neuron Network. <br>
# The training is base on Keras and using tensorflow backend to run.

# ## Import necessary package and do gpu test.

# In[1]:


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


# ## Input data with H5py package
# Input data size: 1 channel image with shape 224x224. <br>
# Input label size: 1 or 0 with onehot encoding. <br>
# \# of input data(train(with val), test): 30000,6000

# In[2]:


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
for i in range(5,8):
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
        for a in range(0,6000):
            test_img_pt.append(pt_[a])
        for b in range(0,6000):
            test_label_.append(label_[b])


# ## Define function to do normalize and duplicate image to 3 channel. <br>
# Image will be normalize to 0~255 and duplicate into 3 channel.

# In[3]:


def normalize_and_rgb(img, label):  
    output = np.stack([np.asanyarray(img)/ np.asanyarray(img).max(), np.asanyarray(img)/ np.asanyarray(img).max(), np.asanyarray(img)/ np.asanyarray(img).max()],axis=-1)
    return output, np.asanyarray(label)
train_img, train_label = normalize_and_rgb(train_img_pt, train_label_)
test_img, test_label = normalize_and_rgb(test_img_pt, test_label_)


# In[4]:


#print(train_img, train_label, test_img, test_label)
#train_label = np.asanyarray(train_label_)
#test_label = np.asanyarray(test_label_)
#print(train_label.shape, test_label.shape)


# ## Construsting a sequential 
# Construst a model with Keras pre-configured ResNet-50 and one Dense layer with Softmax activation function. <br>
# Using Earlystopping and custom callback function to prevent training from overfitting and collect ROC/AUC data point. 

# In[5]:


from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.applications.resnet50 import ResNet50
from keras.callbacks import EarlyStopping,ModelCheckpoint
import keras

model = Sequential()
#model.load_weights=('../aws-fpga-top-tagging-notebooks/dataset_Tom/weights-floatingpoint-224x224-fixval-best/class_weights.h5')
model.add(ResNet50(weights=None, pooling='max'))
model.add(Dense(2, activation='softmax'))


model.summary()
modelcheckpoint = ModelCheckpoint('./model/ResNet-50-best.model',
                                                     monitor='val_acc', save_best_only=True)
earlystop = EarlyStopping(monitor='val_loss', patience=0, mode='min')


# In[6]:


from sklearn.metrics import roc_auc_score
from keras.callbacks import Callback
class roc_callback(Callback):
    def __init__(self,training_data,validation_data):
        self.x = training_data[0]
        self.y = training_data[1]
        self.x_val = validation_data[0]
        self.y_val = validation_data[1]


    def on_train_begin(self, logs={}):
        return

    def on_train_end(self, logs={}):
        return

    def on_epoch_begin(self, epoch, logs={}):
        return

    def on_epoch_end(self, epoch, logs={}):
        y_pred = self.model.predict(self.x)
        roc = roc_auc_score(self.y, y_pred)
        y_pred_val = self.model.predict(self.x_val)
        roc_val = roc_auc_score(self.y_val, y_pred_val)
        print('\rroc-auc: %s - roc-auc_val: %s' % (str(round(roc,4)),str(round(roc_val,4))),end=100*' '+'\n')
        return

    def on_batch_begin(self, batch, logs={}):
        return

    def on_batch_end(self, batch, logs={}):
        return

model.compile(optimizer='adam', loss='binary_crossentropy',metrics=['accuracy'])
train_history = model.fit(x= train_img, y = train_label, validation_split=0.1, 
                          epochs=10, verbose=1, batch_size=4,
                          callbacks=[earlystop, roc_callback(training_data=(train_img, train_label),validation_data=(test_img, test_label))] )
#train_history = model.fit_generator(train_dataset.flow(train_img, train_label, batch_size = 16),\
#                                    validation_data=val_dataset.flow(test_img, test_label, batch_size = 16), epochs=10,\
#                                    verbose=1,use_multiprocessing=True, validation_freq=1)


# ## Define function to visualize learning curve and prediction. <br><br>

# In[9]:


#Define a function to show training history
def show_train_history(train_history, train, validation,filename):
    plt.plot(train_history.history[train])
    plt.plot(train_history.history[validation])
    plt.title('Train history')
    plt.xlabel('Epoch')
    plt.ylabel(train)
    plt.legend(['train','validaiton'],loc='upper left')
    plt.savefig(filename)
    plt.show()

def plot_images_prediction(iamges, labels, prediction, idx, num = 10):
    fig = plt.gcf()
    fig.set_size_inches(12,14)
    if num>25: num=25
    for i in range(0,num):
        ax=plt.subplot(5,5,1+i)
        ax.imshow(iamges[idx], cmap='binary')
        title = "label=" + str(np.argmax(labels[idx]))
        if len(prediction)>0 :
            title += ",prediction=" + str(np.argmax(prediction[idx]))
        ax.set_title(title,fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
        idx+=1
    plt.savefig(filename)
    plt.show()
    
#Show the result of training 
show_train_history(train_history,'accuracy','val_accuracy','./acc.png')
show_train_history(train_history,'loss','val_loss',filename='./loss.png')


# In[10]:


#Scoring the accuracy of model by test dataset.
score = model.evaluate(test_img, test_label)
print()
print("Accuracy of model is", score[1])

#Prediction 
prediction = model.predict(test_img)
print(prediction)

test_label_wo_onehot = np.zeros(len(test_label))
pred = np.zeros(len(test_label))
for i in range(len(test_label)):
    test_label_wo_onehot[i] = np.argmax(test_label[i])
    pred[i] = np.argmax(prediction[i])

plot_images_prediction(test_img,test_label, prediction,idx=1,num=5)




# In[11]:


#Display confusion matrix
pd.crosstab(test_label_wo_onehot, pred, rownames=['label'], colnames=['predict'])


# In[12]:


#Display true value and predict value 
df = pd.DataFrame({'label':test_label_wo_onehot,'predict':pred})
df[:10]


# In[13]:


#Find the result that true value is x but pred value is y (x != y)
df[(df.label==0)&(df.predict==1)]


# In[26]:


from sklearn.metrics import roc_curve,roc_auc_score
 
fpr , tpr , thresholds = roc_curve ( test_label_wo_onehot  , pred )

def plot_roc_curve(fpr,tpr,filename): 
    plt.plot(fpr,tpr)
    plt.axis([0,1,0,1]) 
    plt.xlabel('False Positive Rate') 
    plt.ylabel('True Positive Rate')
    plt.savefig(filename)
    plt.show()    

plot_roc_curve(fpr,tpr,'./roc.png') 
auc_score = roc_auc_score(test_label_wo_onehot, pred)


# In[ ]:




