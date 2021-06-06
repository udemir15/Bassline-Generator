from tensorflow.keras.backend import random_normal, exp
from tensorflow.keras import layers
from tensorflow import shape


class Sampling(layers.Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = shape(z_mean)[0]
        dim = shape(z_mean)[1]
        #original_dim = z_mean.shape[1]
        #latent_dim = z_mean.shape[2]
        epsilon = random_normal(shape=(batch, dim), mean=0., stddev=0.1)
        return z_mean + exp(0.5 * z_log_var) * epsilon
