# -*- coding: utf-8 -*-
"""
Created on Mon Jan 11 01:07:32 2021

@author: onee
"""

from keras.models import Model
from keras.layers import Input, Dense, Activation, Flatten, BatchNormalization
from keras.layers import Conv2D, MaxPooling2D

def MultiCls(x, n_class):
    inputs = Input(shape=x.shape)

    main_network = Conv2D(32, kernel_size=(3,3), padding="same")(inputs)
    main_network = Activation("relu")(main_network)

    main_network = Conv2D(32, kernel_size=(3,3), padding="same")(main_network)
    main_network = Activation("relu")(main_network)

    main_network = MaxPooling2D(pool_size=(2,2))(main_network)

    main_network = Conv2D(64, kernel_size=(3,3), padding="same")(main_network)
    main_network = Activation("relu")(main_network)

    main_network = Conv2D(64, kernel_size=(3,3), padding="same")(main_network)
    main_network = Activation("relu")(main_network)

    main_network = MaxPooling2D(pool_size=(2,2))(main_network)

    main_network = Flatten()(main_network)
    main_network = Dense(128)(main_network)
    main_network = Activation('relu')(main_network)

    main_network = Dense(n_class)(main_network)
    out = Activation('softmax')(main_network)
       
    model = Model(inputs=inputs, outputs=out)

    return model