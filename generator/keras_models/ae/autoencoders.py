from tensorflow.keras import Model
from tensorflow.keras import layers
from keras_models.ae.encoders import *
from keras_models.ae.decoders import *
from sklearn.mixture import GaussianMixture


class AE(Model):
    def __init__(self, name, **kwargs):
        super(AE, self).__init__(name=name, **kwargs)
        self.gmm = None

    def call(self, inputs):
        return self.decoder(self.encoder(inputs))

    def sample(self, num_samples, X=None, n_components=None):
        if X is not None:
            assert n_components != None, "Please insert the number of components to train the Gaussian Mixture Model!"
            num_samples_x = X.shape[0]
            train_encoded = self.encoder(X).numpy()
            self.encoded_dims = train_encoded.shape[1:]
            self.gmm = GaussianMixture(n_components=n_components)
            print("Fitting GMM to distribution of encoding.")
            self.gmm.fit(train_encoded.reshape(num_samples_x, -1))
        assert self.gmm != None, "Please first insert a dataset to train a Gaussian Mixture Model"
        print(self.encoded_dims)
        samples = self.gmm.sample(num_samples)[0].reshape(
            num_samples, *self.encoded_dims)
        return self.decoder(samples).numpy().argmax(-1)


class RNNAE(AE):
    def __init__(self, encoder_hidden_units, decoder_hidden_units, timesteps, vocab_size, building_encoder_rnn="LSTM",
                 building_decoder_rnn="LSTM", embed_size=32, bidirectional_encoder=False, dropout=0,
                 name="RNNEncoderRNNDecoder", **kwargs):
        super(RNNAE, self).__init__(name=name, **kwargs)
        self.encoder = RNNEncoder(encoder_hidden_units, vocab_size, embed_size, bidirectional_encoder, dropout, getattr(
            layers, building_encoder_rnn), name=f"{'Bi' if bidirectional_encoder else ''}{building_encoder_rnn}_Encoder")
        self.decoder = RNNDecoder(decoder_hidden_units, vocab_size, timesteps, dropout, getattr(
            layers, building_decoder_rnn), name=f"{building_decoder_rnn}_Decoder")


class RNNEncoderCNNDecoder(AE):
    def __init__(self, encoder_hidden_units, decoder_filter_sizes, timesteps, vocab_size, decoder_dilations=1,
                 building_encoder_rnn="LSTM", embed_size=32, bidirectional_encoder=False, dropout=0, name="RNNEncoderCNNDecoder",
                 **kwargs):
        super(RNNEncoderCNNDecoder, self).__init__(name=name, **kwargs)
        self.encoder = RNNEncoder(encoder_hidden_units, vocab_size, embed_size, bidirectional_encoder, dropout, getattr(
            layers, building_encoder_rnn), name=f"{'Bi' if bidirectional_encoder else ''}{building_encoder_rnn}_Encoder")
        self.decoder = CNNDecoder(
            decoder_filter_sizes, vocab_size, decoder_dilations, timesteps=timesteps, encoder_type="rnn")


class CNNAE(AE):
    def __init__(self, encoder_filter_sizes, decoder_filter_sizes, vocab_size, encoder_dilations=1, decoder_dilations=1,
                 embed_size=32, name="CNNEncoderCNNDecoder", **kwargs):
        super(CNNAE, self).__init__(name=name, **kwargs)
        self.encoder = CNNEncoder(
            encoder_filter_sizes, encoder_dilations, vocab_size, embed_size)
        self.decoder = CNNDecoder(
            decoder_filter_sizes, vocab_size, decoder_dilations)


class CNNEncoderRNNDecoder(AE):
    def __init__(self, encoder_filter_sizes, decoder_hidden_units, timesteps, vocab_size, building_decoder_rnn="LSTM",
                 encoder_dilations=1, embed_size=32, dropout=0, name="CNNEncoderRNNDecoder", **kwargs):
        super(CNNEncoderRNNDecoder, self).__init__(name=name, **kwargs)
        self.encoder = CNNEncoder(
            encoder_filter_sizes, encoder_dilations, vocab_size, embed_size)
        self.decoder = RNNDecoder(decoder_hidden_units, vocab_size, timesteps, dropout, getattr(
            layers, building_decoder_rnn), encoder_type="cnn", name=f"{building_decoder_rnn}_Decoder")
