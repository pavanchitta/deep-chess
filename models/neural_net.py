import tensorflow as tf
import keras
from keras import layers, models, Input
from models.autoencoder import Autoencoder
import git
import sys
repo = git.Repo("./", search_parent_directories=True)
homedir = repo.working_dir
sys.path.append(f'{homedir}')

import copy



class NeuralNet:

    def __init__(self, input_dim=(773, ), autoenc_model='autoenc_10000_bsize64_epochs50'):
        self.model = self.create_model(input_dim, autoenc_model)

    def create_model(self, input_dim, autoenc_model):

        pre_trained_autoenc = keras.models.load_model(f'{homedir}/models/trained_models/{autoenc_model}')

        left_autoencoder = Autoencoder(input_dim)
        left_input = left_autoencoder.model.input
        left_autoencoder.model.set_weights(pre_trained_autoenc.get_weights())
        # Share weights for left and right autoencoder
        right_autoencoder = Autoencoder(input_dim, name='right')
        right_input = right_autoencoder.model.input
        right_autoencoder.model.set_weights(pre_trained_autoenc.get_weights())
        left_embed_layer = left_autoencoder.model.layers[-2].output
        right_embed_layer = right_autoencoder.model.layers[-2].output
        merge_layer = keras.layers.Concatenate()([left_embed_layer, right_embed_layer])
        dnn_layer = keras.layers.Dense(100, activation='relu')(merge_layer)
        dnn_layer = keras.layers.Dense(100, activation='relu')(dnn_layer)
        output = keras.layers.Dense(2, activation='softmax')(dnn_layer)

        model = keras.models.Model(inputs=[left_input, right_input], outputs=output)
        print(model.summary())

        model.compile(optimizer=keras.optimizers.Adam(),
                      loss=keras.losses.BinaryCrossentropy(),
                      metrics=['accuracy'])
        return model


    def fit(self, left_X, right_X, y, batch_size=64, epochs=100, validate=False):
        if not validate:
            self.model.fit(x=[left_X, right_X], y=y, epochs=epochs, batch_size=batch_size, verbose=1)
        else:
            self.model.fit(x=[left_X, right_X], y=y, epochs=epochs, batch_size=batch_size, validation_split=0.2, verbose=1)

    def predict(self, X):
        # Return the likelihood that the left game is better for white
        return self.model.predict(X)[0][0]


    def evaluate(self, X, y):
        # TODO
        pass
