from tensorflow.keras import layers


class DenseDecoder(layers.Layer):
    def __init__(self, intermediate_dims, timesteps, vocab_size, name="DenseDecoder", **kwargs):
        super(DenseDecoder, self).__init__(name=name, **kwargs)
        if type(intermediate_dims) == int:
            intermediate_dims = (intermediate_dims, )
        self.dense_layers = [layers.Dense(
            intermediate_dim * (timesteps if idx == 0 else 1), activation="relu") for idx, intermediate_dim in enumerate(intermediate_dims)]
        self.reshape = layers.Reshape((timesteps, intermediate_dims[0]))
        self.classifier = layers.Dense(vocab_size, activation="softmax")

    def call(self, inputs):
        x = self.dense_layers[0](inputs)
        x = self.reshape(x)
        for dense_layer in self.dense_layers[1:]:
            x = dense_layer(x)
        return self.classifier(x)


class RNNDecoder(layers.Layer):
    def __init__(self, hidden_units, vocab_size, timesteps=64, dropout=0, building_rnn=layers.LSTM, encoder_type="rnn",
                 name="RNN_Decoder", **kwargs):
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

    def call(self, inputs):
        if self.encoder_type == "rnn":
            x = self.repeat_vector(inputs)
        else:
            x = layers.Reshape(
                (self.timesteps, -1))(inputs)
        for rnn_layer in self.rnn_layers:
            x = rnn_layer(x)
        return self.classifier(x)


class CNNDecoder(layers.Layer):
    def __init__(self, num_filters, dense_unit, vocab_size, dilation_rates, timesteps=None, encoder_type="cnn", name="CNN_Decoder", **kwargs):
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
        if type(num_filters) == int:
            num_filters = (num_filters, )
        if type(dilation_rates) == int:
            dilation_rates = (dilation_rates, ) * len(num_filters)
        self.dense_layer = layers.Dense(
            self.timesteps // (2**(len(num_filters) - 1)) * dense_unit, activation="relu")
        self.reshape_layer = layers.Reshape(
            (self.timesteps // (2**(len(num_filters) - 1)), dense_unit))
        self.conv_layers = [(layers.Conv1DTranspose if encoder_type == "cnn" else layers.Conv1D)(
            num_filter, kernel_size=3, strides=2 if encoder_type == "cnn" and idx != len(num_filters) - 1 else 1,
            padding="same", dilation_rate=dilation_rate, activation="relu")
            for idx, (num_filter, dilation_rate) in enumerate(zip(num_filters, dilation_rates))]
        self.classifier = layers.Dense(vocab_size, activation="softmax")

    def call(self, inputs):
        x = self.dense_layer(inputs)
        x = self.reshape_layer(x)
        for conv_layer in self.conv_layers:
            x = conv_layer(x)
        if self.encoder_type == "rnn":
            x = layers.Reshape(
                (self.timesteps, -1))(x)
        return self.classifier(x)
