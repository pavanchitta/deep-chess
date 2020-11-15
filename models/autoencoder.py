import tensorflow as tf
import keras
from keras import layers, models, Input
import models.data_processing as dp

INPUT_DIM = (773, )

class Autoencoder:

    def __init__(self, input_dim, name=''):
        self.model = self.create_model(input_dim, name)
        self.name = name

    def create_model(self, input_dim=(773, ), name=''):

        input = Input(shape=input_dim)
        dnn = layers.Dense(100, activation='relu')(input)
        dnn = layers.Dense(100, activation='relu')(dnn)
        dnn = layers.Dense(100, activation='relu')(dnn)
        dnn = layers.Dense(100, activation='relu')(dnn)
        output = layers.Dense(input_dim[0], activation='sigmoid')(dnn)
        model = keras.models.Model(inputs=[input], outputs=output, name=name)

        model.compile(optimizer=keras.optimizers.Adam(),
                      loss=keras.losses.BinaryCrossentropy(),
                      metrics=['accuracy'])

        return model

    def fit(self, X, batch_size=64, epochs=100, validate=False):
        if not validate:
            self.model.fit(X, X, epochs=epochs, batch_size=batch_size, verbose=1)
        else:
            self.model.fit(X, X, epochs=epochs, batch_size=batch_size, validation_split=0.2, verbose=1)

    def predict(self, X):
        return self.model.predict(X)

    def evaluate(self, X):
        # TODO
        pass


if __name__ == "__main__":

    epochs = 50
    batch_size = 64
    ngames = 10000

    X = dp.load_autoencoder_data(name='', ngames=ngames)
    autoencoder = Autoencoder(input_dim=INPUT_DIM)
    autoencoder.fit(X, epochs=epochs, batch_size=batch_size, validate=True)
    autoencoder.model.save(f'trained_models/autoenc_{ngames}_bsize{batch_size}_epochs{epochs}')

