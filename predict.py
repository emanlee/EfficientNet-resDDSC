from __future__ import print_function
import keras
import keras as ks
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import SGD
from keras.callbacks import EarlyStopping,ModelCheckpoint
import os,sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn import metrics
from scipy import interp


import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.layers import Input, Conv2D, DepthwiseConv2D, BatchNormalization, ReLU, Add, GlobalAveragePooling2D, Dense
from tensorflow.keras.models import Model

from tensorflow.keras.layers import Input, Conv2D, Dense, Flatten, BatchNormalization
from tensorflow.keras.layers import GlobalAveragePooling2D, Add, Multiply, Reshape
from tensorflow.nn import swish


import keras
import keras as ks
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import SGD, Adam
from keras.callbacks import EarlyStopping,ModelCheckpoint
import os,sys
import numpy as np
import matplotlib
import argparse
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn import metrics
from scipy import interp
################################


# Setup argument parser
parser = argparse.ArgumentParser(description="This script processes RNA-seq expression data to generate predictions using a trained model.")
parser.add_argument('-length_TF', type=int, help='Number of data parts divided')
parser.add_argument('-data_path', type=str, help='Path to the 2Dhistogram_data')
parser.add_argument('-num_classes', type=int, help='Number of label classes')
parser.add_argument('-model_path', type=str, help='Path to the trained model file')

# Parse arguments
args = parser.parse_args()

def load_data_TF2(indel_list,data_path):
    import random
    import numpy as np
    xxdata_list = []
    yydata = []
    count_set = [0]
    count_setx = 0
    for i in indel_list:#len(h_tf_sc)):
        xdata = np.load(data_path+'/Nxdata_tf' + str(i) + '.npy')
        for k in range(xdata.shape[0]):
            xxdata_list.append(xdata[k,:,:,:])
        count_setx = count_setx + xdata.shape[0]
        count_set.append(count_setx)
    return((np.array(xxdata_list),count_set))


length_TF = args.length_TF
data_path = args.data_path
num_classes = args.num_classes
model_path = args.model_path
print ('select', type)
whole_data_TF = [i for i in range(length_TF)]
test_TF = [i for i in range (length_TF)]
(x_test, count_set) = load_data_TF2(test_TF,data_path)
print(x_test.shape, 'x_test samples')
############



def SEBlock(input, ratio=4):
    x = GlobalAveragePooling2D()(input)
    x = Dense(input.shape[-1] // ratio, activation='relu')(x)
    x = Dense(input.shape[-1], activation='sigmoid')(x)
    x = Reshape((1, 1, input.shape[-1]))(x)
    return Multiply()([input, x])

def conv2d_bn(inpt, filters=64, kernel_size=(3,3), strides=1, padding='same'):
    #卷积、归一化和relu三合一
    x = ks.layers.Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, padding=padding)(inpt)
    x = ks.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    return x

def basic_bottle(inpt, filters=64, kernel_size=(3,3), strides=1, padding='same', if_baisc=False):
    #18中的4个basic_bottle
    x = conv2d_bn(inpt, filters=filters, kernel_size=kernel_size, strides=strides, padding=padding)
    x = conv2d_bn(x, filters=filters)
    if if_baisc==True:
        temp = conv2d_bn(inpt, filters=filters, kernel_size=(1,1), strides=2, padding='same')
        outt = ks.layers.add([x, temp])
    else:
        outt = ks.layers.add([x, inpt])
    return outt


def Bottleneck1(input, filters, kernel, t, s, r=False, dilation_rate=1):
    # Depth

    x = Conv2D(filters * t, (1, 1), padding='same')(input)
    x = BatchNormalization()(x)
    x = swish(x)

    # Width

    x = ks.layers.DepthwiseConv2D(kernel_size=kernel, strides=(s, s), dilation_rate=(dilation_rate, dilation_rate), padding='same')(
        x)
    x = BatchNormalization()(x)
    x = swish(x)

    # SE
    x = SEBlock(x)

    # Depth
    x = Conv2D(filters, (1, 1), padding='same')(x)
    x = BatchNormalization()(x)
    # x = ks.layers.Dropout(rate=0.2,
    #                       noise_shape=(None, 1, 1, 1))(x)
    if r:
        x = Add()([x, input])
    return x


def Bottleneck3(input, filters, kernel, t, s, r=False, dilation_rate= 3):
    # Depth

    x = Conv2D(filters * t, (1, 1), padding='same')(input)
    x = BatchNormalization()(x)
    x = swish(x)

    # Width

    x = ks.layers.DepthwiseConv2D(kernel_size=kernel, strides=(s, s), dilation_rate=(dilation_rate, dilation_rate), padding='same')(
        x)
    x = BatchNormalization()(x)
    x = swish(x)

    # SE
    x = SEBlock(x)

    # Depth
    x = Conv2D(filters, (1, 1), padding='same')(x)
    x = BatchNormalization()(x)
    x = ks.layers.Dropout(rate=0.2,
                          noise_shape=(None, 1, 1, 1))(x)

    if r:
        x = Add()([x, input])
    return x


def EfficientNetB0(xx):
    input = ks.layers.Input(xx.shape[1:])
    x = Conv2D(32, (3, 3), strides=(2, 2), padding='same')(input)
    x = BatchNormalization()(x)
    x = swish(x)
    x = conv2d_bn(x, filters=16, kernel_size=(7, 7), strides=2, padding='valid')
    x2 = ks.layers.MaxPool2D(pool_size=(3, 3), strides=2)(x)
    x2 = basic_bottle(x2, filters=16, kernel_size=(3, 3), strides=1, padding='same', if_baisc=False)
    x2 = basic_bottle(x2, filters=16, kernel_size=(3, 3), strides=1, padding='same', if_baisc=False)
    x2 = Bottleneck1(x2, 16, (3, 3), 1, 1)
    x2 = Bottleneck1(x2, 32, (5, 5), 6, 2)
    x2 = Bottleneck1(x2, 32, (5, 5), 6, 1, True)
    x2 = Bottleneck1(x2, 64, (3, 3), 6, 2)
    x2 = Bottleneck1(x2, 64, (3, 3), 6, 1, True)
    x2 = Bottleneck3(x2, 128, (3, 3), 6, 2)
    x2 = Bottleneck3(x2, 128, (3, 3), 6, 1, True)
    x2 = Bottleneck3(x2, 256, (5, 5), 6, 2)
    x2 = Bottleneck3(x2, 256, (5, 5), 6, 1, True)
    x2 = Bottleneck1(x2, 320, (3, 3), 6, 1)
    x = Conv2D(1280, (1, 1), padding='same')(x2)
    x = BatchNormalization()(x)
    x = swish(x)

    x = GlobalAveragePooling2D()(x)
    output = Dense(3, activation='softmax')(x)

    model = Model(input, output)
    return model


model = EfficientNetB0(x_test)
sgd = SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])


###########################################################3
save_dir = os.path.join(os.getcwd(),'a1_pdx322_predict_results_no_y_den1')
model.load_weights(model_path)
if not os.path.isdir(save_dir):
    os.makedirs(save_dir)
print ('load model and predict')
y_predict = model.predict(x_test)
#np.save(save_dir+'/y_test.npy',y_test)
np.save(save_dir+'/a1_pdx322_y_predict_den1.npy',y_predict)
s = open (save_dir+'/a1_pdx322_gene_index_den1.txt','w')
for i in count_set:
    s.write(str(i)+'\n')
s.close()
######################################