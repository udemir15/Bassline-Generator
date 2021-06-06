from tensorflow.keras import layers
from models.layers import ConvMaxPooling1D


class RNNEncoder(layers.Layer):
    def __init__(self, hidden_units, vocab_size, embed_size=32, bidirectional=False, dropout=0,
                 building_rnn=layers.LSTM, name="LSTM_Encoder", **kwargs):
        super(RNNEncoder, self).__init__(name=name, **kwargs)
        assert type(hidden_units) == int or type(
            hidden_units) == tuple, "Please enter a tuple of hidden_units or an integer"
        self.embedding_layer = layers.Embedding(vocab_size, embed_size)
        if type(hidden_units) == int:
            hidden_units = (hidden_units, )
        self.rnn_layers = [building_rnn(
            hidden_unit, return_sequences=True, dropout=dropout) for hidden_unit in hidden_units[:-1]]
        self.rnn_layers.append(building_rnn(hidden_units[-1], dropout=dropout))
        if bidirectional:
            self.rnn_layers = [layers.Bidirectional(
                rnn_layer) for rnn_layer in self.rnn_layers]

    def call(self, inputs):
        x = self.embedding_layer(inputs)
        for rnn_layer in self.rnn_layers:
            x = rnn_layer(x)
        return x


class CNNEncoder(layers.Layer):
    def __init__(self, num_filters, dilations, vocab_size, embed_size=32, name="Conv_Encoder", **kwargs):
        if type(num_filters) == tuple and type(dilations) == tuple:
            assert len(num_filters) == len(
                dilations), "num_filters and dilations must have same length if both tuple"
        super(CNNEncoder, self).__init__(name=name, **kwargs)
        self.embedding_layer = layers.Embedding(vocab_size, embed_size)
        if type(num_filters) == int:
            num_filters = (num_filters, )
        if type(dilations) == int:
            dilations = (dilations, ) * len(num_filters)
        self.convmaxpooling1d_layers = [ConvMaxPooling1D(num_filter, kernel_size=3, dilation=dilation, activation="relu")
                                        for num_filter, dilation in zip(num_filters, dilations)]

    def call(self, inputs):
        x = self.embedding_layer(inputs)
        for convmaxpooling1d_layer in self.convmaxpooling1d_layers:
            x = convmaxpooling1d_layer(x)
        return x
