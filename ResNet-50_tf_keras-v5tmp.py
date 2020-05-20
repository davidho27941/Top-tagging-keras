#!/usr/bin/env python
# coding: utf-8

# # Top tagging model with ResNet-50 

# This is a training script about using top tagging data to train a ResNet-50 Neuron Network. <br>
# The training is base on Keras and using tensorflow backend to run.

# ## Import necessary package and do gpu test.

# In[2]:


import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd 
import hdf5plugin, h5py, os, glob, datetime, tables, time, threading, cython
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


# ## Construsting a sequential 
# Construst a model with Keras pre-configured ResNet-50 and one Dense layer with Softmax activation function. <br>
# Using Earlystopping and custom callback function to prevent training from overfitting and collect ROC/AUC data point. 

# In[3]:


"""
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
"""
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
    return model

model = create_model()
plot_model(model, to_file='./Picture/Sequential_Model.png')

Image('./Picture/Sequential_Model.png')


# ## Input data with H5py package
# Input data size: 1 channel image with shape 224x224. <br>
# Input label size: 1 or 0 with onehot encoding. <br>
# \# of input data(train(with val), test): 30000,6000

# In[3]:


prefix = "/home/david/Storage/dataset_Tom/img224_all/converted/rotation_224_v1/"
train_filename = np.asanyarray(glob.glob(os.path.join(prefix,"train_*")))
test_filename = np.asanyarray(glob.glob(os.path.join(prefix,"test_*")))
val_filename = np.asanyarray(glob.glob(os.path.join(prefix,"val_*")))
print(type(train_filename), len(train_filename))
print(type(test_filename), len(test_filename))
print(type(val_filename), len(val_filename))


# In[4]:


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
    r = threading.Thread( target=remove_data(os.path.join(prefix,file.join(seq))) )
    r.start()
    r.join
    os.path.join(os.path.join(prefix,file.join(seq)))
    t = threading.Thread( target = write_data(os.path.join(os.path.join(prefix,file.join(seq))), im, lal, mode) )
    t.start()
    t.join()
    tEnd = time.time()
    print("Pre-processing duration: {0}s".format(tEnd - tStart))
    return int(int(len(im))/Batch)

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
#def train_model(x,y):
#    logdir = os.path.join("logs","fit", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"),)
#    tensorboard_callback = tf.keras.callbacks.TensorBoard(logdir, histogram_freq=1, write_graph=True, write_images=True, )
#    train_history = model.fit(x, y, epochs=1, batch_size=64, shuffle=True, callbacks=[tensorboard_callback])
#    #model.summary()
#    return train_history, model




# In[5]:


auc = []
val_auc = []
loss = []
val_loss = []
batch_zise = 32

#logdir = os.path.join("logs","fit", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"),)
#tensorboard_callback = tf.keras.callbacks.TensorBoard(logdir, histogram_freq=1, write_graph=False, write_images=False, )

#train stage      

def train_process(round):
    j=0
    if ( (i+1) % 3 != 0 or ((i+1) % 3 == 0 and (i+1) <= 6)):
        STEP_train = Preprocessing(train_filename[i],"train", batch_zise)
        image_train = tfio.IODataset.from_hdf5(filename=os.path.join(prefix,'tmp_train.h5'), dataset="/image")
        label_train = tfio.IODataset.from_hdf5(filename=os.path.join(prefix,'tmp_train.h5'), dataset="/label")
        train = tf.data.Dataset.zip((image_train,label_train)).batch(batch_zise, drop_remainder=True).prefetch(tf.data.experimental.AUTOTUNE)
        print("+--------------------------------------------------------------------------------------------+")
        print("Round {0}".format(i+1))
        train_history = model.fit(train, epochs=1, steps_per_epoch=STEP_train,  shuffle=True)
        with open('train_log.txt', 'r+') as f:
            f.write("{0:.3f}, {1:.3f}\n".format(train_history.history['auc'], train_history.history['auc']))
    if ( (i+1) % 3 == 0 and (i+1) > 6  ):
        STEP_train = Preprocessing(train_filename[i],"train", batch_zise)
        STEP_val = Preprocessing(val_filename[j],"val", batch_zise)
        image_train = tfio.IODataset.from_hdf5(filename=os.path.join(prefix,'tmp_train.h5'), dataset="/image")
        label_train = tfio.IODataset.from_hdf5(filename=os.path.join(prefix,'tmp_train.h5'), dataset="/label")
        image_val = tfio.IODataset.from_hdf5(filename=os.path.join(prefix,'tmp_val.h5'), dataset="/image")
        label_val = tfio.IODataset.from_hdf5(filename=os.path.join(prefix,'tmp_val.h5'), dataset="/label")
        train = tf.data.Dataset.zip((image_train,label_train)).batch(batch_zise, drop_remainder=True).prefetch(tf.data.experimental.AUTOTUNE)
        val = tf.data.Dataset.zip((image_val,label_val)).batch(batch_zise, drop_remainder=True).prefetch(tf.data.experimental.AUTOTUNE)
        print("+--------------------------------------------------------------------------------------------+")
        print("Round {0}".format(i+1))
        train_history = model.fit(train, validation_data=val, epochs=1, steps_per_epoch=STEP_train, validation_steps=STEP_val,  shuffle=True)
        with open('train_log.txt', 'r+') as f:
            f.write("{0:.3f}, {1:.3f}\n".format(train_history.history['auc'], train_history.history['auc']))
        with open('train_log_val.txt', 'r=') as g:
            g.write("{0:.3f}, {1:.3f}\n".format(train_history.history['val_auc'], train_history.history['val_loss']))
        j +=1

for i in range(len(train_filename)):
    train_threading = threading.Thread(job=train_process(i))
    train_threading.start()
    train_threading.join


# In[ ]:


STEP_train


# ## Define function to visualize learning curve and prediction. <br><br>

# In[ ]:


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


    
#Show the result of training 
show_train_history(train_history,'loss','val_loss','./loss_new.png')
show_train_history(train_history,'auc','val_auc',filename='./auc_new.png')


# In[14]:


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


# In[15]:


def plot_roc_curve(fpr,tpr,filename): 
    x = np.linspace(0,1)
    y = x
    plt.plot(fpr,tpr,label='model')
    plt.plot(x,y,label='random')
    plt.axis([0,1,0,1]) 
    plt.xlabel('False Positive Rate') 
    plt.ylabel('True Positive Rate')
    plt.legend(loc='lower right')
    plt.savefig(filename)
    plt.show() 


# In[16]:


from sklearn.metrics import roc_curve,roc_auc_score

#Scoring the accuracy of model by test dataset.
score = model.evaluate(test_img, test_label)
print()
print("Accuracy of model is", score[1])


#Prediction 
prediction = model.predict(test_img)
print(prediction)

plot_images_prediction(test_img,test_label, prediction,idx=1,num=5)


modelname = ""



if (score[1] > 0.85):
    path = os.path.join("model","best",)
    seq = (path, "my_weight_best_new_{0}.h5".format(datetime.datetime.now().strftime("%Y%m%d-%H%M%S")))
    model.save_weights(modelname.join(seq))
    model.save('./model/my_model_best_new_{0}.h5'.format(datetime.datetime.now().strftime("%Y%m%d-%H%M%S")))
    print('A best model was saved.')
    fpr , tpr , thresholds = roc_curve ( test_label[:,1]  , prediction[:,1] )   
    plot_roc_curve(fpr,tpr,'./roc_new_best_{0}.png'.format(datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))) 
    auc_score = roc_auc_score(test_label[:,1]  , prediction[:,1])
    print(auc_score)
elif (score[1] > 0.75) and (score[1] < 0.85):
    path = os.path.join("model","better",datetime.datetime.now().strftime("%Y%m%d-%H%M%S"),)
    seq = (path, "my_weight_better_new.h5")
    model.save_weights(modelname.join(seq))
    model.save('./model/my_model_better_new.h5')
    print('A better model was saved.')
    fpr , tpr , thresholds = roc_curve (test_label[:,1]  , prediction[:,1] )
    plot_roc_curve(fpr,tpr,'./roc_new_better.png') 
    auc_score = roc_auc_score(test_label[:,1]  , prediction[:,1])
    print(auc_score)
else:
    print('Not a good model, please try again.')



# In[17]:


#Display confusion matrix
#pd.crosstab(test_label_wo_onehot, pred, rownames=['label'], colnames=['predict'])


# In[ ]:


#Display true value and predict value 
df = pd.DataFrame({'label':test_label_wo_onehot,'predict':pred})
df[:10]
#Find the result that true value is x but pred value is y (x != y)
#df[(df.label==1)&(df.predict==0)]


# ## Loading pre-trained model and test 

# In[18]:


from sklearn.metrics import roc_curve,roc_auc_score
from tensorflow.keras.models import load_model
model_best = load_model('model/my_model_best_new_20200308-174931.h5')
model_best.summary()

model_best.load_weights('model/bestmy_weight_best_new_20200308-174931.h5')





# In[19]:


best_prediction = model_best.predict(test_img)
#predict_class = model_best.predict_class(test_img)
best_score = model_best.evaluate(test_img, test_label,batch_size=64, steps=10)
print()
print("Accuracy of model is", best_score[1])

#plot_images_prediction(test_img, test_label, predict_class,idx=1,num=5)
best_fpr , best_tpr , best_thresholds = roc_curve (test_label[:,1], best_prediction[:,1])   
plot_roc_curve(best_fpr, best_tpr,'./roc_best.png') 
best_auc_score = roc_auc_score(test_label[:,1], best_prediction[:,1])
print(best_auc_score)


# In[20]:


kwargs = dict(histtype='stepfilled', alpha=0.3, normed=True, bins=100)
plt.hist(best_prediction[:,0],**kwargs, label='top')
plt.hist(best_prediction[:,1],**kwargs,label='QCD')
plt.yscale('log')
plt.xlim(0.0,1.0)
plt.legend(loc='upper right')


# In[ ]:


qq, c1, c2, tt = 0, 0, 0, 0 
for i in range(len(img)): 
    if (np.argmax(label[i]) == 1):
        qq += 255*img[i]/img[i].max()
        c1 += 1
    else :
        tt += 255*img[i]/img[i].max()
        c2 +=1
print("c1: {0}, c2: {1}".format(c1,c2))
plt.figure()        
color_map = plt.imshow(qq)
color_map.set_cmap("Blues_r")
plt.colorbar()
plt.xlabel(r'$i\eta$')
plt.ylabel(r'$i\phi$')
plt.title('QCD, average 15K events')
plt.savefig("qcd.png")

plt.figure()
color_map = plt.imshow(tt)
color_map.set_cmap("Blues_r")
plt.colorbar()
plt.xlabel(r'$i\eta$')
plt.ylabel(r'$i\phi$')
plt.title('Top, average 15K events')
plt.savefig("top.png")
qq, c1, c2, tt = 0, 0, 0, 0 

