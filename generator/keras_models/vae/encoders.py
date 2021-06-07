from tensorflow.keras import layers
from keras_models.vae.utils import Sampling
from keras_models.layers import ConvMaxPooling1D


class DenseEncoder(layers.Layer):

    def __init__(self, intermediate_dims, latent_dim, vocab_size, embed_size=32, name="DenseEncoder", **kwargs):
        super(DenseEncoder, self).__init__(name=name, **kwargs)
        self.embedding = layers.Embedding(vocab_size, embed_size)
        if type(intermediate_dims) == int:
            intermediate_dims = (intermediate_dims, )
        self.dense_layers = [layers.Dense(
            intermediate_dim, activation="relu") for intermediate_dim in intermediate_dims]
        self.flatten = layers.Flatten()
        self.z_mean_layer = layers.Dense(latent_dim)
        self.z_log_sigma_layer = layers.Dense(latent_dim)
        self.sampling = Sampling()

    def call(self, inputs):
        x = self.embedding(inputs)
        for dense_layer in self.dense_layers:
            x = dense_layer(x)
        x = self.flatten(x)
        z_mean = self.z_mean_layer(x)
        z_log_sigma = self.z_log_sigma_layer(x)
        z = self.sampling((z_mean, z_log_sigma))
        return z_mean, z_log_sigma, z


class RNNEncoder(layers.Layer):

    def __init__(self, hidden_units, last_dense_dim, latent_dim, vocab_size, embed_size=32, bidirectional=False, dropout=0,
                 building_rnn=layers.LSTM, name="RNN_Encoder", **kwargs):
        super(RNNEncoder, self).__init__(name=name, **kwargs)
        assert type(hidden_units) == int or type(
            hidden_units) == tuple, "Please enter a tuple of hidden_units or an integer"
        self.embedding = layers.Embedding(vocab_size, embed_size)
        if type(hidden_units) == int:
            hidden_units = (hidden_units, )
        self.rnn_layers = [building_rnn(hidden_unit, return_sequences=idx != len(
            hidden_units) - 1, dropout=dropout) for idx, hidden_unit in enumerate(hidden_units)]
        if bidirectional:
            self.rnn_layers = [layers.Bidirectional(
                rnn_layer) for rnn_layer in self.rnn_layers]
        self.last_dense_layer = layers.Dense(last_dense_dim, activation='relu')
        self.z_mean_layer = layers.Dense(latent_dim)
        self.z_log_sigma_layer = layers.Dense(latent_dim)
        self.sampling = Sampling()

    def call(self, inputs):
        x = self.embedding(inputs)
        for rnn_layer in self.rnn_layers:
            x = rnn_layer(x)
        x = self.last_dense_layer(x)
        z_mean = self.z_mean_layer(x)
        z_log_sigma = self.z_log_sigma_layer(x)
        z = self.sampling((z_mean, z_log_sigma))
        return z_mean, z_log_sigma, z


class CNNEncoder(layers.Layer):

    def __init__(self, num_filters, dilation_rates, vocab_size, latent_dim, embed_size=32, name="Conv_Encoder", **kwargs):
        if type(num_filters) == tuple and type(dilation_rates) == tuple:
            assert len(num_filters) == len(
                dilation_rates), "num_filters and dilations must have same length if both tuple"
        super(CNNEncoder, self).__init__(name=name, **kwargs)
        self.embedding_layer = layers.Embedding(vocab_size, embed_size)
        if type(num_filters) == int:
            num_filters = (num_filters, )
        if type(dilation_rates) == int:
            dilation_rates = (dilation_rates, ) * len(num_filters)
        self.convmaxpooling1d_layers = [ConvMaxPooling1D(num_filter, kernel_size=3, dilation_rate=dilation_rate)
                                        for num_filter, dilation_rate in zip(num_filters, dilation_rates)]
        self.flatten = layers.Flatten()
        self.z_mean_layer = layers.Dense(latent_dim)
        self.z_log_sigma_layer = layers.Dense(latent_dim)
        self.sampling = Sampling()

    def call(self, inputs):
        x = self.embedding_layer(inputs)
        for convmaxpooling1d_layer in self.convmaxpooling1d_layers:
            x = convmaxpooling1d_layer(x)
        x = self.flatten(x)
        z_mean = self.z_mean_layer(x)
        z_log_sigma = self.z_log_sigma_layer(x)
        z = self.sampling((z_mean, z_log_sigma))
        return z_mean, z_log_sigma, z
