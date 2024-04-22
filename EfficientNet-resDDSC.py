from __future__ import print_function



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
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn import metrics
import argparse
from scipy import interp
####################################### parameter settings
data_augmentation = False
# num_predictions = 20
batch_size = 1024 #mini batch for training
#num_classes = 3   #### categories of labels
epochs = 20    #### iterations of trainning, with GPU 1080, 600 for KEGG and Reactome, 200 for tasks for GTRD
#length_TF =3057  # number of divide data parts
# num_predictions = 20
model_name = 'den_trained_model.h5'
###################################################


# Setup argument parser
parser = argparse.ArgumentParser(description="This script processes RNA-seq expression data to generate models and perform evaluations.")
parser.add_argument('-length_TF', type=int, help='Number of data parts divided')
parser.add_argument('-dataset_path', type=str, help='Path to the 2Dhistogram_data')
parser.add_argument('-num_classes', type=int, help='Number of label classes')
parser.add_argument('-output_path', type=str, required=True, help='Path for the output text file')

args = parser.parse_args()


def load_data_TF2(indel_list,data_path): # cell type specific  ## random samples for reactome is not enough, need borrow some from keggp
    import random
    import numpy as np
    xxdata_list = []
    yydata = []
    count_set = [0]
    count_setx = 0
    for i in indel_list:#len(h_tf_sc)):
        xdata = np.load(data_path+'/Nxdata_tf' + str(i) + '.npy')
        ydata = np.load(data_path+'/ydata_tf' + str(i) + '.npy')
        for k in range(len(ydata)):
            xxdata_list.append(xdata[k,:,:,:])
            yydata.append(ydata[k])
        count_setx = count_setx + len(ydata)
        count_set.append(count_setx)
        print (i,len(ydata))
    yydata_array = np.array(yydata)
    yydata_x = yydata_array.astype('int')
    print (np.array(xxdata_list).shape)
    return((np.array(xxdata_list),yydata_x,count_set))

# if len(sys.argv) < 4:
#     print ('No enough input files')
#     sys.exit()
length_TF =int(args.length_TF) # number of data parts divided
data_path = args.dataset_path
num_classes = int(args.num_classes)
whole_data_TF = [i for i in range(length_TF)]




def SEBlock(input, ratio=4):
    x = GlobalAveragePooling2D()(input)
    x = Dense(input.shape[-1] // ratio, activation='relu')(x)
    x = Dense(input.shape[-1], activation='sigmoid')(x)
    x = Reshape((1, 1, input.shape[-1]))(x)
    return Multiply()([input, x])

def conv2d_bn(inpt, filters=64, kernel_size=(3,3), strides=1, padding='same'):

    x = ks.layers.Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, padding=padding)(inpt)
    x = ks.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    return x

def basic_bottle(inpt, filters=64, kernel_size=(3,3), strides=1, padding='same', if_baisc=False):

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

sys.stdout = open(args.output_path, 'w')
###################################################################################################################################
for test_indel in range(1,4): ################## three fold cross validation                                                     ## for  3 fold CV
    test_TF = [i for i in range (int(np.ceil((test_indel-1)*0.333*length_TF)),int(np.ceil(test_indel*0.333*length_TF)))]         #
    train_TF = [i for i in whole_data_TF if i not in test_TF]                                                                    #
###################################################################################################################################
#####################################################################
    (x_train, y_train,count_set_train) = load_data_TF2(train_TF,data_path)
    (x_test, y_test,count_set) = load_data_TF2(test_TF,data_path)
    print(x_train.shape, 'x_train samples')
    print(x_test.shape, 'x_test samples')
    save_dir = os.path.join(os.getcwd(),str(test_indel)+'saved_models'+str(epochs)) ## the result folder
    if num_classes >2:
        y_train = keras.utils.to_categorical(y_train, num_classes)
        y_test = keras.utils.to_categorical(y_test, num_classes)
    print(y_train.shape, 'y_train samples')
    print(y_test.shape, 'y_test samples')
    ############
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    if num_classes < 3:
        print('no enough categories')
        sys.exit()
    else:
        model = EfficientNetB0(x_train)
        sgd = SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
        model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])

    early_stopping = keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=50, verbose=0, mode='auto')
    checkpoint1 = ModelCheckpoint(filepath=save_dir + '/weights.{epoch:02d}-{val_loss:.2f}.hdf5',
                                  monitor='val_loss',
                                  verbose=1, save_best_only=False, save_weights_only=False, mode='auto', period=1)
    checkpoint2 = ModelCheckpoint(filepath=save_dir + '/weights.hdf5', monitor='val_accuracy', verbose=1,
                                  save_best_only=True, mode='auto', period=1)
    callbacks_list = [checkpoint2, early_stopping]
    if not data_augmentation:
        print('Not using data augmentation.')
        history = model.fit(x_train, y_train,
                            batch_size=batch_size,
                            epochs=epochs, validation_split=0.2,
                            shuffle=True, callbacks=callbacks_list)

    # Save model and weights

    model_path = os.path.join(save_dir, model_name)
    model.save(model_path)
    print('Saved trained model at %s ' % model_path)
    # Score trained model.
    scores = model.evaluate(x_test, y_test, verbose=1)
    print('Test loss:', scores[0])
    print('Test accuracy:', scores[1])
    y_predict = model.predict(x_test)
    np.save(save_dir + '/end_y_test.npy', y_test)
    np.save(save_dir + '/end_y_predict.npy', y_predict)
    ############################################################################## plot training process
    plt.figure(figsize=(10, 6))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.grid()
    plt.legend(['train', 'val'], loc='upper left')
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.grid()
    plt.savefig(save_dir + '/result.pdf')
    ###############################################################  evaluation without consideration of data separation
    if num_classes == 3:  ## here we only focus on three category tasks
        plt.figure(figsize=(10, 6))
        for i in range(3):
            y_test_x = [j[i] for j in y_test]
            y_predict_x = [j[i] for j in y_predict]
            fpr, tpr, thresholds = metrics.roc_curve(y_test_x, y_predict_x, pos_label=1)
            plt.subplot(1, 3, i + 1)
            plt.plot(fpr, tpr)
            plt.grid()
            plt.plot([0, 1], [0, 1])
            plt.xlabel('FP')
            plt.ylabel('TP')
            plt.ylim([0, 1])
            plt.xlim([0, 1])
            auc = np.trapz(tpr, fpr)
            print('AUC:', auc)
            plt.title('label' + str(i) + ', AUC:' + str(auc))
        plt.savefig(save_dir + '/end_3labels.pdf')
        #######################################
        plt.figure(figsize=(10, 6))
        y_predict1 = []
        y_test1 = []
        x = 2
        for i in range(int(len(y_predict) / 3)):
            y_predict1.append(y_predict[3 * i][x] - y_predict[3 * i + 1][
                x])  #### here we prepared the data as (GeneA,GeneB),(GeneB,GeneA) and (GeneA,GeneX) as label 1, 2, 0, That is why we can predict direaction using this code
            y_predict1.append(-y_predict[3 * i][x] + y_predict[3 * i + 1][x])
            y_test1.append(y_test[3 * i][x])
            y_test1.append(y_test[3 * i + 1][x])
        fpr, tpr, thresholds = metrics.roc_curve(y_test1, y_predict1, pos_label=1)
        # Print ROC curve
        plt.plot(fpr, tpr)
        plt.plot([0, 1], [0, 1])
        # Print AUC
        auc = np.trapz(tpr, fpr)
        print('AUC:', auc)
        plt.ylim([0, 1])
        plt.xlim([0, 1])
        plt.grid()
        plt.title('AUC:' + str(auc))
        plt.xlabel('FP')
        plt.ylabel('TP')
        plt.savefig(save_dir + '/end.pdf')
        #############################################################
        ################################################ evaluation with data separation

        fig = plt.figure(figsize=(5, 5))
        plt.plot([0, 1], [0, 1])
        plt.ylim([0, 1])
        plt.xlim([0, 1])
        plt.xlabel('FP')
        plt.ylabel('TP')
        # plt.grid()
        AUC_set = []
        y_testy = y_test
        y_predicty = y_predict
        tprs = []
        mean_fpr = np.linspace(0, 1, 6)
        s = open(save_dir + '/AUCs.txt', 'w')
        for jj in range(len(count_set) - 1):  # len(count_set)-1):
            if count_set[jj] < count_set[jj + 1]:
                print(jj, count_set[jj], count_set[jj + 1])
                y_test = y_testy[count_set[jj]:count_set[jj + 1]]
                y_predict = y_predicty[count_set[jj]:count_set[jj + 1]]
                y_predict1 = []
                y_test1 = []
                x = 2
                for i in range(int(len(y_predict) / 3)):
                    y_predict1.append(y_predict[3 * i][x] - y_predict[3 * i + 1][x])
                    y_predict1.append(-y_predict[3 * i][x] + y_predict[3 * i + 1][x])
                    y_test1.append(y_test[3 * i][x])
                    y_test1.append(y_test[3 * i + 1][x])
                fpr, tpr, thresholds = metrics.roc_curve(y_test1, y_predict1, pos_label=1)
                tprs.append(np.interp(mean_fpr, fpr, tpr))
                tprs[-1][0] = 0.0
                plt.plot(fpr, tpr, color='0.5', lw=0.1)
                auc = np.trapz(tpr, fpr)
                s.write(
                    str(jj) + '\t' + str(count_set[jj]) + '\t' + str(count_set[jj + 1]) + '\t' + str(auc) + '\n')
                print('AUC:', auc)
                AUC_set.append(auc)
        mean_tpr = np.median(tprs, axis=0)
        mean_tpr[-1] = 1.0
        per_tpr = np.percentile(tprs, [25, 50, 75], axis=0)
        mean_auc = np.trapz(mean_tpr, mean_fpr)
        plt.plot(mean_fpr, mean_tpr, 'k', lw=3, label='median ROC')
        plt.title(str(mean_auc))
        plt.fill_between(mean_fpr, per_tpr[0, :], per_tpr[2, :], color='g', alpha=.2, label='Quartile')
        plt.legend(loc='lower right')
        plt.savefig(save_dir + '/ROCs_percentile.pdf')
        del fig
        fig = plt.figure(figsize=(5, 5))
        plt.hist(AUC_set, bins=50)
        plt.savefig(save_dir + '/ROCs_hist.pdf')
        del fig
        s.close()
    sys.stdout.close()



