import tables
import h5py 
import numpy as np
import matplotlib.pyplot as plt 
import hdf5plugin
import os
import glob
import tqdm
import tensorflow as tf 
import time 
from sklearn.metrics import roc_auc_score
from tensorflow.keras.utils import plot_model
import tensorflow_io as tfio 
from IPython.display import Image
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
import datetime
from sklearn.metrics import roc_curve,roc_auc_score

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  # Restrict TensorFlow to only use the first GPU
    try:
        tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
        tf.config.experimental.set_virtual_device_configuration(
        gpus[0],
        [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=20000)])
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU")
    except RuntimeError as e:
    # Visible devices must be set before GPUs have been initialized
        print(e)

# Define normalization function
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

class hdf5_data_generator():
    """
    Custom class for creating a pipeline.
    
    === Parameters ===
    data_path: String. Directory of data files. The files must be hdf5 format.
    length: Int. Number of records in a files.
    batch_size: Int. Batch size.
    number of files: String. Specify how many files are there exist in the `data_path`.
    shuffle: Boolean. To shuffle the sequence of dataset.
    """
    def __init__(self, data_path, length, batch_size, num_of_files, shuffle):
        self.data_path = data_path
        self.length = length
        self.batch_size = batch_size
        self.num_of_files = num_of_files
        self.shuffle = shuffle
        self.idx = np.arange(self.length)
    def __len__(self):
        """
        This is a object to clarify the steps per epoch.
        """
        return int(np.floor(self.num_of_files*(self.length) / self.batch_size))
    def pipeline(self):
        """
        This is a object to pass the data to trainging task from files.
        """
        while True:
            if self.shuffle:
                np.random.shuffle(self.data_path)
            else:
                pass
            for i in range(self.num_of_files):
#                 print(f"Now using: {self.data_path[i]}")
                with h5py.File(self.data_path[i], 'r') as file:
                    img, label = normalize_and_rgb(file['img_pt'][:], file['label'][:])
                if self.shuffle:
                    np.random.shuffle(self.idx)
                    img, label = img[self.idx], label[self.idx]
                for j in range(0, self.length-self.batch_size, self.batch_size):
                    yield (img[j:j+self.batch_size], label[j:j+self.batch_size])

# Setting log directory 
logdir = "logs/" + datetime.now().strftime("%Y%m%d-%H%M%S")

# Create tensorboard callback function.
tensorborad_callback = tf.keras.callbacks.TensorBoard(
                        log_dir=logdir, 
                        histogram_freq=1, 
                        write_graph=True,
                        write_images=False, 
                        update_freq='epoch', 
                        profile_batch=2,
                    )

# Setting hyper-parameters
BATCH_SIZE = 64
number_of_train_file = 20
number_of_test_file = 3
number_of_val_file = 3
events_per_file = 10000
EPOCH = 50

# Setting paths
PREFIX = "/home/david/workplace/Data_from_Tom/img224_all/converted/rotation_224_v1/"
TRAIN_LIST = np.sort(np.array(glob.glob(PREFIX + "train_file*.h5")))
TEST_LIST = np.sort(np.array(glob.glob(PREFIX + "test_file*.h5")))
VAL_LIST = np.sort(np.array(glob.glob(PREFIX + "val_file*.h5")))

# Create data object for training/ testing/ validating.
train = hdf5_data_generator(TRAIN_LIST, 
                            events_per_file, 
                            BATCH_SIZE, 
                            number_of_train_file, 
                            shuffle=True)

test = hdf5_data_generator(TEST_LIST, 
                           events_per_file, 
                           BATCH_SIZE, 
                           number_of_test_file, 
                           shuffle=True)

val = hdf5_data_generator(VAL_LIST, 
                          events_per_file, 
                          BATCH_SIZE, 
                          number_of_val_file, 
                          shuffle=True)


# Define model 
def create_resnet_50():
    model = tf.keras.models.Sequential([
    tf.keras.applications.ResNet50V2(weights=None, pooling='max', classes=2048),
    tf.keras.layers.Dense(1024, activation='relu'),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(2, activation='softmax'),
    ])
    model.compile(optimizer='adam', 
                  loss='categorical_crossentropy', 
                  metrics=[tf.keras.metrics.AUC(num_thresholds=200, curve='ROC')])
    model.summary()
    model.save('./model/')
    model.save('./model/tmp_model.h5')
    return model

resnet_50 = create_resnet_50()

# Training
train_history = resnet_50.fit(train.pipeline(), 
                              validation_data=val.pipeline(), 
                              epochs=EPOCH, 
                              verbose=1, 
                              steps_per_epoch=len(train), 
                              validation_steps=len(val), 
                              validation_freq=2, 
                              callbacks=tensorborad_callback,
                             )

# Save model structure and weights
resnet_50.save('./model/tmp_model.h5')
resnet_50.save_weights('./model/tmp_weight.h5')


# Show learning curve 
plt.figure(figsize=(8,6))
x_axis = np.arange(len(train_history.history['loss']))
x_axis_val = np.arange(len(train_history.history['val_loss']))
plt.plot(x_axis, train_history.history['loss'], label='train loss')
plt.plot(x_axis, train_history.history['auc'], label='train auc')
plt.plot(x_axis_val, train_history.history['val_loss'], label='val loss')
plt.plot(x_axis_val, train_history.history['val_auc'], label='val auc')
plt.legend(loc='upper right')
plt.title("Training_history")
plt.xlabel("Epoch")
plt.ylabel("Performance")
plt.savefig("Learning_curve.png")

# Scoring the accuracy of model by test dataset.
score = resnet_50.evaluate(test.pipeline(), steps=len(test), verbose=1)
print()
print("Accuracy of model is", score[1])


# Show prediction 
with h5py.File(TEST_LIST[0], 'r') as file:
    test_img, test_label = normalize_and_rgb(file['img_pt'][:], file['label'][:])

prediction = resnet_50.predict(test_img)
print(f"First 10 prediction and ground truth.")
print(f"Prediction: {prediction[:10]}, ground truth: {test_label[:10]}")

# plot_images_prediction(test_img, test_label, prediction,idx=1,num=5)


modelname = ""

# Ranking the model performace and save it
if (score[1] > 0.85):
    path = os.path.join("model","best",)
    seq = (path, "my_weight_best_new_{0}.h5".format(datetime.datetime.now().strftime("%Y%m%d-%H%M%S")))
    resnet_50.save_weights(modelname.join(seq))
    resnet_50.save('./model/my_model_best_new_{0}.h5'.format(datetime.datetime.now().strftime("%Y%m%d-%H%M%S")))
    print('A best model was saved.')
    fpr , tpr , thresholds = roc_curve( test_label[:,1]  , prediction[:,1] )   
    plot_roc_curve(fpr,tpr,'./roc_new_best_{0}.png'.format(datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))) 
    auc_score = roc_auc_score(test_label[:,1]  , prediction[:,1])
    print(auc_score)
elif (score[1] > 0.75) and (score[1] < 0.85):
    path = os.path.join("model","better",datetime.datetime.now().strftime("%Y%m%d-%H%M%S"),)
    seq = (path, "my_weight_better_new.h5")
    resnet_50.save_weights(modelname.join(seq))
    resnet_50.save('./model/my_model_better_new.h5')
    print('A better model was saved.')
    fpr , tpr , thresholds = roc_curve(test_label[:,1]  , prediction[:,1] )
    plot_roc_curve(fpr,tpr,'./roc_new_better.png') 
    auc_score = roc_auc_score(test_label[:,1]  , prediction[:,1])
    print(auc_score)
else:
    print('Not a good model, please try again.')