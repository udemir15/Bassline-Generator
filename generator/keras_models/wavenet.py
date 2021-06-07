import numpy as np
from tensorflow.keras import layers, models


class WaveNET(models.Model):
    def __init__(self, num_filters, vocab_size, embed_size=32, name="WaveNET", **kwargs):
        super(WaveNET, self).__init__(name=name, **kwargs)
        self.embedding_layer = layers.Embedding(vocab_size, embed_size)
        # (2 ** np.arange(6)).astype(int).tolist()
        dilations = np.ones(6).astype(int).tolist()
        if type(num_filters) == int:
            num_filters = (num_filters, )
        self.conv_layers = [layers.Conv1D(num_filter, kernel_size=2, dilation_rate=dilation, activation="relu", padding='same')
                            for num_filter, dilation in zip(num_filters, dilations)]
        self.classifier = layers.Dense(vocab_size, activation='softmax')

    def call(self, inputs):
        x = self.embedding_layer(inputs)
        for conv_layer in self.conv_layers:
            x = conv_layer(x)
        return self.classifier(x)
