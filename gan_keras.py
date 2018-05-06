from __future__ import print_function
import sys
import os
import math
import random
import time
import pickle
import collections
import keras
import tqdm
import gc

import pandas as pd
import numpy as np
import keras
import warnings

from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import LSTM,Flatten,Dense,Embedding,GRU,BatchNormalization,Dropout
from keras.layers.convolutional import Conv1D,Conv2D,UpSampling2D
from keras.layers.convolutional import MaxPooling1D,MaxPooling2D
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau

import matplotlib.pyplot as plt
import seaborn as sns

warnings.simplefilter("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def load_data():
    img_rows, img_cols = 28, 28
    (X_train, y_train), (X_test, y_test) = mnist.load_data()

    # channels first

    X_train = X_train.reshape(X_train.shape[0], 1, img_rows, img_cols)
    X_test = X_test.reshape(X_test.shape[0], 1, img_rows, img_cols)

    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')

    X_train /= 255
    X_test /= 255
    data = {'train':X_train,
    'test':X_test}
    return data

def plotter(data,n_images = 36):
    X_train = data['train']    
    dim = (np.sqrt(n_images),np.sqrt(n_images))
    
    plt.figure(figsize=(10,8))
    
    indices = [random.randint(0,n_images) for i in range(n_images)]
    for i in range(len(indices)):
        plt.subplot(dim[0],dim[1],i+1)
        img = X_train[indices[i],0]
        plt.imshow(img, cmap="Greys")
        plt.axis("off")
    plt.show()

# Freeze weights in the discriminator for stacked training

def make_trainable(net, val):

    net.trainable = val
    for l in net.layers:
        l.trainable = val
    return net

def generator(data):

    print("\nBuilding the Generator Network")
    X_train = data['train']
    X_test = data['test']
    shp = X_train.shape[1:]
    dropout_rate = 0.25
    opt = keras.optimizers.Adam(lr=0.01,
        decay=1e-5)
    # generative model
    model = Sequential()
    
    
    
    nch = 256
    model.add(Dense(nch*14*14,kernel_initializer='glorot_normal',activation = 'relu',input_shape = (100,)))
    model.add(keras.layers.core.Reshape( [nch, 14, 14] ))
    model.add(UpSampling2D((2,2),data_format = 'channels_first'))
    
    model.add(Conv2D(nch//2,(3,3),
        data_format = 'channels_first',
        kernel_initializer = 'glorot_normal',
        padding = 'same',
        activation = 'relu'))
    
    model.add(Conv2D(nch//4,(3,3),
        kernel_initializer = 'glorot_normal',
        data_format = 'channels_first',
        padding = 'same',
        activation = 'relu'))
    model.add(Conv2D(nch//4,(3,3),
        kernel_initializer = 'glorot_normal',
        data_format = 'channels_first',
        padding = 'same',
        activation = 'relu'))
    
    model.add(MaxPooling2D(pool_size=(2, 2),
        data_format = 'channels_first',
        padding = 'same'))
    model.add(UpSampling2D((2,2),data_format = 'channels_first'))
    model.add(Conv2D(1,(1,1),
        kernel_initializer = 'glorot_normal',
        data_format = 'channels_first',
        padding = 'same',
        activation = 'relu'))
    
    model.add(keras.layers.LeakyReLU(alpha=0.3))
    model.add(UpSampling2D((2,2),data_format = 'channels_first'))
    model.add(MaxPooling2D(pool_size=(2,2),
        data_format = 'channels_first',
        padding = 'same'))
        
    model.add(BatchNormalization())
    model.compile(loss='binary_crossentropy', optimizer=opt)
    #model.summary()
    return model

def discriminator(data):
    print("\nBuilding the Discriminative Network")
    X_train = data['train']
    X_test = data['test']
    shp = X_train.shape[1:]
    dropout_rate = 0.25
    dopt = keras.optimizers.Adam(lr=0.009,
        decay=1e-5)
    # Build Discriminative model ...
    #d_input = keras.layers.Input(shape=shp)
    discriminator = Sequential()
    discriminator.add(Conv2D(256, (5, 5),
        strides =(2, 2), 
        padding = 'same',
        input_shape = shp,
        data_format = "channels_first"))
    discriminator.add(keras.layers.LeakyReLU(0.2))
    discriminator.add(Dropout(dropout_rate))
    discriminator.add(Conv2D(512, (3, 3),
        data_format="channels_first",
        strides=(2, 2),
        activation="relu", 
        padding="same"))
    discriminator.add(keras.layers.LeakyReLU(0.2))
    discriminator.add(Dropout(dropout_rate))
    discriminator.add(Flatten())
    discriminator.add(Dense(256,activation = 'relu'))
    discriminator.add(Dropout(dropout_rate))
    discriminator.add(Dense(2,activation='softmax'))
    discriminator.compile(loss='categorical_crossentropy', optimizer=dopt)
    #discriminator.summary()

    discriminator = make_trainable(discriminator, False)
    return discriminator

def GAN(data,d,g):
    
    print("\nBuilding the GAN Network")

    opt = keras.optimizers.Adam(lr=0.01,
        decay=1e-5)

    
    # creating the GAN network
    gan = Sequential()
    gan.add(g)
    gan.add(d)
    gan.compile(loss = 'categorical_crossentropy',optimizer = opt)
    #gan.summary()
    return gan
def plot_temp(arr,name = None):
    print(name)#"real MNIST images")
    plt.figure(1)
    dim=(4,4)
    for i in range(16):
        plt.subplot(dim[0],dim[1],i+1)
        img = arr[i,0,:,:]
        plt.imshow(img, cmap="Greys")
        plt.axis('off')
    plt.tight_layout()
    plt.show()

def plot_loss(losses):
    IPython.display.clear_output(wait=True)
    IPython.display.display(plt.gcf())
    plt.figure(figsize=(10,8))
    plt.plot(losses["d"], label='discriminitive loss')
    plt.plot(losses["g"], label='generative loss')
    plt.legend()
    plt.show()

def plot_gen(n_epoch = None, epochs = None,g = None,n_ex=9,dim=(3,3), figsize=(8,8)):
    
    noise = np.random.uniform(0,1,size=[n_ex,100])
    generated_images = g.predict(noise)

    plt.figure(figsize=figsize)
    for i in range(generated_images.shape[0]):
        plt.subplot(dim[0],dim[1],i+1)
        img = generated_images[i,0,:,:]        
        plt.imshow(img, cmap="Greys")
        plt.axis('off')
    plt.tight_layout()
    plt.savefig(os.getcwd()+os.sep+'results'+os.sep+"{0}_epochs_{1}.png".format(epochs,n_epoch))
    plt.show()
def train_for_n(data,nb_epoch=5000, plt_frq=25,BATCH_SIZE=32):

    X_train, X_test = data['train'],data['test']
    count = 0
    g = generator(data)#model
    d = discriminator(data)#model
    gan = GAN(data,d,g)

    
    
    # generate some images
    ntrain = 12000
    trainidx = random.sample(range(0, X_train.shape[0]), ntrain)
    XT = X_train[trainidx,:,:,:]

    # Pre-train the discriminator network ...
    # on some generated data

    noise_gen = np.random.uniform(0,1,size=[XT.shape[0],100])
    generated_images = g.predict(noise_gen)

    X = np.concatenate((XT, generated_images))
    losses = {"d":[], "g":[]}
    n = XT.shape[0]
    y = np.zeros([2*n,2])
    y[:n,1] = 1
    y[n:,0] = 1
    ### getting discriminator performance upto
    ### this point
    d.fit(X,y, verbose = 2,nb_epoch=2, batch_size=256)
    y_hat = d.predict(X)
    y_hat_idx = np.argmax(y_hat,axis=1)
    y_idx = np.argmax(y,axis=1)
    diff = y_idx-y_hat_idx
    n_tot = y.shape[0]
    n_rig = (diff==0).sum()
    acc = n_rig*100.0/n_tot
    print("\nAccuracy: {} pct ({} of {}) right".format(acc, n_rig, n_tot))    
    # plot_temp(XT,name = 'real images')
    # plot_temp(generated_images,name = "generated images")
    del acc,n_rig,n_tot,diff,y_hat,y_idx,y_hat_idx
    gc.collect()
    make_trainable(d,True)
    for e in tqdm.tqdm(range(1, nb_epoch)):
        
        # Make generative images
        image_batch = X_train[np.random.randint(0,X_train.shape[0],size=BATCH_SIZE),:,:,:]    
        noise_gen = np.random.uniform(0,1,size=[BATCH_SIZE,100])
        generated_images = g.predict(noise_gen)
        
        # Train discriminator on generated images
        X = np.concatenate((image_batch, generated_images))
        y = np.zeros([2*BATCH_SIZE,2])
        y[0:BATCH_SIZE,1] = 1
        y[BATCH_SIZE:,0] = 1
        
        make_trainable(d,True)
        d_loss  = d.train_on_batch(X,y)
        losses["d"].append(d_loss)
    
        # train Generator-Discriminator stack on input noise to non-generated output class
        noise_tr = np.random.uniform(0,1,size=[BATCH_SIZE,100])
        y2 = np.zeros([BATCH_SIZE,2])
        y2[:,1] = 1
        
        make_trainable(d,False)
        g_loss = gan.train_on_batch(noise_tr, y2 )
        losses["g"].append(g_loss)
        
        if count%plt_frq == 0:
            # Updates plots
            plot_loss(losses)
            #plot_gen(g = g,epochs = e+1, n_epoch = nb_epoch)
        count += 1
    return losses

data = load_data()

losses = train_for_n(data,nb_epoch=5000, plt_frq=5,BATCH_SIZE=32)
plot_loss(losses)