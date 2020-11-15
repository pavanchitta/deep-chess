####
# Train and save a model
####

import numpy as np
import data_processing as dp
import neural_net


if __name__ == "__main__":

    epochs = 50
    batch_size = 64
    ngames = 10000
    NAME='test'

    X, y = dp.load_neural_net_data(name='', ngames=ngames)
    print("X shape: ", X.shape)
    print("Y shape: ", y.shape)

    nn_mdl = neural_net.NeuralNet(input_dim=(773,))

    nn_mdl.fit(X[:, 0, :], X[:, 1, :], y, batch_size=batch_size, epochs=epochs, validate=True)

    nn_mdl.model.save(f'trained_models/{NAME}_{ngames}_bsize{batch_size}_epochs{epochs}')
