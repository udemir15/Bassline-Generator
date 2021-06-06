from datetime import datetime, time
from numpy.random import normal
from pandas import DataFrame
import tensorflow as tf
from tensorflow.keras import models
from tensorflow.keras import layers
from tensorflow.keras.metrics import Mean
from tensorflow.keras.losses import sparse_categorical_crossentropy
from keras_models.vae.encoders import *
from keras_models.vae.decoders import *


class VAE(models.Model):
    def __init__(self, name="VAE", **kwargs):
        super(VAE, self).__init__(name=name, **kwargs)
        self.total_loss_tracker = Mean(name="total_loss")
        self.reconstruction_loss_tracker = Mean(name="reconstruction_loss")
        self.kl_loss_tracker = Mean(name="kl_loss")

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
        ]

    def train_step(self, data):
        x, y = data
        if not self.built:
            self.build(x.shape)
        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder(x)
            reconstruction = self.decoder(z)
            reconstruction_loss = tf.reduce_mean(tf.reduce_sum(
                sparse_categorical_crossentropy(y, reconstruction), axis=1))
            kl_loss = -0.5 * (1 + z_log_var -
                              tf.square(z_mean) - tf.exp(z_log_var))
            kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
            total_loss = reconstruction_loss + kl_loss
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }

    def test_step(self, data):
        x, y = data
        z_mean, z_log_var, z = self.encoder(x, training=False)
        reconstruction = self.decoder(z, training=False)
        reconstruction_loss = tf.reduce_mean(tf.reduce_sum(
            sparse_categorical_crossentropy(y, reconstruction), axis=1))
        kl_loss = -0.5 * (1 + z_log_var -
                          tf.square(z_mean) - tf.exp(z_log_var))
        kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
        total_loss = reconstruction_loss + kl_loss
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }

    def call(self, inputs):
        return self.decoder(self.encoder(inputs)[2])

    def sample(self, num_samples=10):
        return self.decoder(normal(0, 0.1, (num_samples, self.latent_dim))).numpy().argmax(axis=-1)

    def save_samples(self, num_samples=10):
        samples = DataFrame(self.sample(num_samples))
        now = '_'.join(str(datetime.now()).split('.')[0].split(' '))
        samples.to_csv(
            f'generations/{len(samples)}_{self.name}_samples_{now}.csv', index=False)


class DenseVAE(VAE):
    def __init__(self, encoder_intermediate_dims, latent_dim, decoder_intermediate_dims, vocab_size, embed_size=16, timesteps=64, name="DenseVAE", **kwargs):
        super(DenseVAE, self).__init__(name=name, **kwargs)
        self.latent_dim = latent_dim
        self.timesteps = timesteps
        self.encoder = DenseEncoder(
            encoder_intermediate_dims, latent_dim, vocab_size, embed_size)
        self.decoder = DenseDecoder(
            decoder_intermediate_dims, timesteps, vocab_size)


class RNNVAE(VAE):
    def __init__(self, encoder_hidden_units, latent_dim, decoder_hidden_units, timesteps, vocab_size, building_encoder_rnn="LSTM",
                 building_decoder_rnn="LSTM", embed_size=32, bidirectional_encoder=False, dropout=0,
                 name="RNNVAE", **kwargs):
        super(RNNVAE, self).__init__(name=name, **kwargs)
        self.latent_dim = latent_dim
        self.timesteps = timesteps
        self.encoder = RNNEncoder(encoder_hidden_units, latent_dim, vocab_size, embed_size, bidirectional_encoder, dropout, getattr(
            layers, building_encoder_rnn), name=f"{'Bi' if bidirectional_encoder else ''}{building_encoder_rnn}_Encoder")
        self.decoder = RNNDecoder(decoder_hidden_units, vocab_size, timesteps, dropout, getattr(
            layers, building_decoder_rnn), name=f"{building_decoder_rnn}_Decoder")


class CNNVAE(VAE):
    def __init__(self, encoder_filter_sizes, latent_dim, decoder_filter_sizes, decoder_dense_unit, timesteps,
                 vocab_size, encoder_dilation_rates=1, decoder_dilation_rates=1, embed_size=16, name="CNNVAE", **kwargs):
        super(CNNVAE, self).__init__(name=name, **kwargs)
        self.latent_dim = latent_dim
        self.timesteps = timesteps
        self.encoder = CNNEncoder(
            encoder_filter_sizes, encoder_dilation_rates, vocab_size, latent_dim, embed_size)
        self.decoder = CNNDecoder(
            decoder_filter_sizes, decoder_dense_unit, vocab_size, decoder_dilation_rates, timesteps=timesteps)
