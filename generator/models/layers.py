from tensorflow.keras import layers


class ConvMaxPooling1D(layers.Layer):
    def __init__(self, filter_size, kernel_size=3, padding="same", dilation_rate=1, activation="relu",
                 name="ConvMaxPooling1D", **kwargs):
        super(ConvMaxPooling1D, self).__init__(name=name, **kwargs)
        self.conv_layer = layers.Conv1D(
            filter_size, kernel_size, dilation_rate=dilation_rate, activation=activation, padding="same")
        self.max_pool_layer = layers.MaxPooling1D(2, padding=padding)

    def call(self, inputs):
        return self.max_pool_layer(self.conv_layer(inputs))


class ConvUpSampling1D(layers.Layer):
    def __init__(self, filter_size, kernel_size=3, padding="same", dilation_rate=1, activation="relu",
                 name="ConvUpSampling1D", **kwargs):
        super(ConvUpSampling1D, self).__init__(name=name, **kwargs)
        self.conv_layer = layers.Conv1D(
            filter_size, kernel_size, dilation_rate=dilation_rate, activation=activation, padding=padding)
        self.up_sampling_layer = layers.UpSampling1D(2)

    def call(self, inputs):
        return self.up_sampling_layer(self.conv_layer(inputs))
