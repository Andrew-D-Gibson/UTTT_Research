import tensorflow as tf


class ResidualUnit(tf.keras.layers.Layer):
    def __init__(self, filters, strides=1, activation='relu', **kwargs):
        super().__init__(**kwargs)
        self.activation = tf.keras.activations.get(activation)
        self.main_layers = [
            tf.keras.layers.Conv2D(filters, (3,3), strides=strides, padding='same', use_bias=False),
            tf.keras.layers.BatchNormalization(),
            self.activation,
            tf.keras.layers.Conv2D(filters, (3,3), strides=1, padding='same', use_bias=False),
            tf.keras.layers.BatchNormalization()
        ]

        self.skip_layers = []
        if strides > 1:
            self.skip_layers = [
                tf.keras.layers.Conv2D(filters, (1,1), strides=strides, padding='same', use_bias=False),
                tf.keras.layers.BatchNormalization()
            ]

    def call(self, inputs):
        Z = inputs 
        for layer in self.main_layers:
            Z = layer(Z)

        skip_Z = inputs 
        for layer in self.skip_layers:
            skip_Z = layer(skip_Z)

        # Note that when layers is empty (when stride is 1 and we're not changing dimensionality)
        # skip_Z remains the input. This is where ResNet's magic happens.

        return self.activation(Z + skip_Z)



class NetworkArchitectureTester:
    @staticmethod
    def convNet():
        ultimate_tic_tac_toe_input = tf.keras.layers.Input(shape=(9,9,4), name='uttt_input')
        
        conv_1 = tf.keras.layers.Conv2D(256, (3,3), padding='same', activation='relu', name='conv_1')(ultimate_tic_tac_toe_input)
        batchnorm = tf.keras.layers.BatchNormalization()(conv_1)
        conv_2 = tf.keras.layers.Conv2D(128, (3,3), strides=(3, 3), activation='relu', name='conv_2')(batchnorm)
        batchnorm = tf.keras.layers.BatchNormalization()(conv_2)
        
        flatten = tf.keras.layers.Flatten()(batchnorm)
        
        dense_1 = tf.keras.layers.Dense(512, activation='relu', name='dense_1')(flatten)
        dense_2 = tf.keras.layers.Dense(256, activation='relu', name='dense_2')(dense_1)
        
        policy_output = tf.keras.layers.Dense(81, activation='softmax', name='policy_output')(dense_2)
        value_output = tf.keras.layers.Dense(1, activation='tanh', name='value_output')(dense_2)
        
        model = tf.keras.models.Model(inputs=ultimate_tic_tac_toe_input, outputs=[policy_output, value_output], name='convNet')

        # Compile model
        losses = {
            'policy_output': 'categorical_crossentropy', 
            'value_output': 'mse'
        }
        
        # 'accuracy' only makes sense for the softmax policy head; the tanh-activated
        # scalar value head would be scored as an (essentially meaningless) exact-match
        # check on a continuous value, so it's only requested for policy_output here.
        model.compile(optimizer='Adam', loss=losses, metrics={'policy_output': 'accuracy'})
        return model


    @staticmethod
    def resNet():
        ultimate_tic_tac_toe_input = tf.keras.layers.Input(shape=(9,9,4), name='uttt_input')
        conv = tf.keras.layers.Conv2D(64, (3,3), padding='same', activation='relu')(ultimate_tic_tac_toe_input)
        batchnorm = tf.keras.layers.BatchNormalization()(conv)

        resLayer = batchnorm
        prev_filters = 64
        for filters in [64]*3 + [128]*3:
            strides = 1 if filters == prev_filters else 3
            resLayer = ResidualUnit(filters, strides=strides)(resLayer)
            prev_filters = filters

        flatten = tf.keras.layers.Flatten()(resLayer)
        
        policy_output = tf.keras.layers.Dense(81, activation='softmax', name='policy_output')(flatten)
        value_output = tf.keras.layers.Dense(1, activation='tanh', name='value_output')(flatten)

        model = tf.keras.models.Model(inputs=ultimate_tic_tac_toe_input, outputs=[policy_output, value_output], name='resNet')

        # Compile model
        losses = {
            'policy_output': 'categorical_crossentropy', 
            'value_output': 'mse'
        }
        
        # 'accuracy' only makes sense for the softmax policy head; the tanh-activated
        # scalar value head would be scored as an (essentially meaningless) exact-match
        # check on a continuous value, so it's only requested for policy_output here.
        model.compile(optimizer='Adam', loss=losses, metrics={'policy_output': 'accuracy'})
        return model

    @staticmethod
    def showModels():
        NetworkArchitectureTester.convNet()
        NetworkArchitectureTester.resNet()