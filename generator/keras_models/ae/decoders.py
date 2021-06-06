from tensorflow.keras import layers
from models.layers import ConvUpSampling1D


class RNNDecoder(layers.Layer):
    def __init__(self, hidden_units, vocab_size, timesteps=64, dropout=0, building_rnn=layers.LSTM, encoder_type="rnn", name="LSTM_Decoder", **kwargs):
        assert encoder_type in [
            "rnn", "cnn"], 'encoder_type must be "rnn" or "cnn"'
        super(RNNDecoder, self).__init__(name=name, **kwargs)
        self.timesteps = timesteps
        self.encoder_type = encoder_type
        if encoder_type == "rnn":
            self.repeat_vector = layers.RepeatVector(timesteps)
        if type(hidden_units) == int:
            hidden_units = (hidden_units, )
        self.rnn_layers = [building_rnn(
            hidden_unit, return_sequences=True, dropout=dropout) for hidden_unit in hidden_units]
        self.classifier = layers.Dense(vocab_size, activation="softmax")

 


class CNNDecoder(layers.Layer):
    def __init__(self, num_filters, vocab_size, dilation_rates, timesteps=None, encoder_type="cnn", name="CNN_Decoder", **kwargs):
        assert encoder_type in [
            "rnn", "cnn"], 'encoder_type must be "rnn" or "cnn"'
        if encoder_type == "rnn":
            assert timesteps != None, "If encoder is rnn timesteps must be supplied!"
        if type(num_filters) == tuple and type(dilation_rates) == tuple:
            assert len(num_filters) == len(
                dilation_rates), "num_filters and dilations must have same length if both tuple"
        super(CNNDecoder, self).__init__(name=name, **kwargs)
        self.timesteps = timesteps
        self.encoder_type = encoder_type
        if encoder_type == "rnn":
            self.repeat_vector = layers.RepeatVector(timesteps)
        if type(num_filters) == int:
            num_filters = (num_filters, )
        if type(dilation_rates) == int:
            dilation_rates = (dilation_rates, ) * len(num_filters)
        self.convupsampling1d_layers = [(layers.Conv1DTranspose if encoder_type == "cnn" else layers.Conv1D)(
            num_filter, kernel_size=3, padding="same", dilation_rate=dilation_rate, activation="relu") for num_filter, dilation_rate in zip(num_filters, dilation_rates)]
        self.last_conv = layers.Conv1D(1, 1, padding='same')
        self.classifier = layers.Dense(vocab_size, activation="softmax")

    def call(self, inputs):
        if self.encoder_type == "rnn":
            x = self.repeat_vector(inputs)
            x = self.convupsampling1d_layers[0](x)
        else:
            x = self.convupsampling1d_layers[0](inputs)
        for convupsampling1d_layer in self.convupsampling1d_layers[1:]:
            x = convupsampling1d_layer(x)
        x = self.last_conv(x)
        if self.encoder_type == "rnn":
            x = layers.Reshape(
                (self.timesteps, -1))(x)
        return self.classifier(x)
