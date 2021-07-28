import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.keras import layers, activations, Sequential, Model
from functools import partial
from tensorflow.keras import backend as K


def roundup(num, divisor=8):
    if num % divisor != 0:
        return (num // divisor + 1) * divisor
    else:
        return num


@tf.function
def hard_swish(x):
    return K.switch(K.less_equal(x, -3), K.zeros_like(x), K.switch(K.greater_equal(x, 3), x, tf.divide(x * tf.add(x, 3.), 6.)))


class SE(layers.Layer):
    """
    Squeeze-and-Excitation Networks
    https://arxiv.org/abs/1709.01507
    """

    def __init__(self,
                 in_channels,
                 reduction_ratio,
                 activation1=activations.relu,
                 activation2=activations.sigmoid):
        super(SE, self).__init__()
        self.pool = layers.GlobalAvgPool2D()
        self.dense1 = layers.Dense(roundup(in_channels // reduction_ratio), activation=activation1)
        self.mult = layers.Multiply()
        self.activation2 = activation2

    def build(self, input_shape):
        self.dense2 = layers.Dense(input_shape[-1], activation=self.activation2)

    def compute_output_shape(self, input_shape):
        return input_shape

    def call(self, inputs, **kwargs):
        out = self.pool(inputs)
        out = self.dense1(out)
        out = self.dense2(out)
        return self.mult([inputs, out])


class ClassicResidual(layers.Layer):
    def __init__(self,
                 filters,
                 kernel_size=3,
                 strides=1,
                 activation=partial(tf.nn.leaky_relu, alpha=.1),
                 se=0,
                 stochastic_depth=0.,
                 norm=True):
        super(ClassicResidual, self).__init__()
        if strides != 1:
            self.shortcut = Sequential([
                layers.AvgPool2D(pool_size=strides, padding="same"),
                layers.Conv2D(filters=filters, kernel_size=1, use_bias=not norm)
            ])
            if norm:
                self.shortcut.add(layers.LayerNormalization(epsilon=1e-5))
        else:
            self.shortcut = lambda x: x

        # stochastic depth: [inputs, residual]
        self.add = layers.Add() if stochastic_depth > 0 else tfa.layers.StochasticDepth(1 - stochastic_depth)

        self.residual = Sequential([
            layers.Conv2D(filters=filters, kernel_size=kernel_size, use_bias=not norm, strides=strides, padding="same")
        ])
        if norm:
            self.residual.add(layers.LayerNormalization(epsilon=1e-5))
        self.residual.add(layers.Activation(activation))
        self.residual.add(layers.Conv2D(filters=filters, kernel_size=kernel_size, use_bias=not norm, padding="same"))
        if norm:
            self.residual.add(layers.LayerNormalization(epsilon=1e-5))
        self.se = se
        self.strides = strides
        self.filters = filters
        self.norm = norm
        self.activation = activation

    def build(self, input_shape):
        if self.se > 0:
            self.residual.add(SE(in_channels=input_shape[-1], reduction_ratio=self.se, activation1=self.activation))
        if self.strides == 1 and self.filters != input_shape[-1]:
            self.shortcut = Sequential([
                layers.Conv2D(filters=self.filters, kernel_size=1, use_bias=not self.norm, padding="same")
            ])
            if self.norm:
                self.shortcut.add(layers.LayerNormalization(epsilon=1e-5))
        del self.se, self.norm, self.strides, self.filters

    def compute_output_shape(self, input_shape):
        return self.residual.compute_output_shape(input_shape)

    def call(self, inputs, *args, **kwargs):
        x = self.residual(inputs)
        shortcut = self.shortcut(inputs)
        x = self.add([shortcut, x])
        return self.activation(x)


def repeat(model,
           filters,
           kernel_size=3,
           strides=1,
           activation=partial(tf.nn.leaky_relu, alpha=.1),
           se=0,
           stochastic_depth=0.,
           norm=True,
           n=2):
    if strides != 1:
        model.add(layers.TimeDistributed(ClassicResidual(filters, kernel_size,
                                                         strides, activation, se,
                                                         stochastic_depth, norm)))
        n -= 1
    for _ in range(n):
        model.add(layers.TimeDistributed(ClassicResidual(filters, kernel_size,
                                                         1, activation, se,
                                                         stochastic_depth, norm)))

def branch(x, shared, v, n, activation):
    x = tfa.layers.NoisyDense(192)(x)
    x = layers.Concatenate()([x, shared])
    x = layers.Activation(activation)(x)
    x =  tfa.layers.NoisyDense(n)(x)
    return layers.Lambda(lambda val: val[0] - tf.reduce_mean(val[0], axis=-1, keepdims=True) + val[1])([x, v])


def get_model(in_shape, frames, out_shape, norm=True, reduced=False, no_se=False):
    model = Sequential()
    model.add(layers.TimeDistributed(layers.Conv2D(32, 3, 2, padding="same", use_bias=not norm),
                                     input_shape=(frames,) + in_shape))
    if norm:
        model.add(layers.TimeDistributed(layers.LayerNormalization(epsilon=1e-5)))
    model.add(layers.Activation(hard_swish))
    repeat(model, 24, strides=1, n=1, norm=norm)
    repeat(model, 32, kernel_size=3 if reduced else 5, strides=2, n=2, norm=norm)
    repeat(model, 40, strides=2, se=0 if no_se else 2, n=2 if reduced else 3, norm=norm)
    repeat(model, 80, strides=2, activation=hard_swish, n=2 if reduced else 4, norm=norm)
    repeat(model, 96, strides=1, se=0 if no_se else 4, activation=hard_swish, n=1 if reduced else 2, norm=norm)
    repeat(model, 128, strides=2, se=0 if no_se else 4, activation=hard_swish,
           n=2 if reduced else 3, norm=norm)
    model.add(layers.Conv3D(160 if reduced else 192, (3, 1, 1), use_bias=not norm))
    if norm:
        model.add(layers.LayerNormalization(epsilon=1e-5))
    model.add(layers.Activation(hard_swish))
    model.add(layers.Conv3D(192 if reduced else 224, (2, 1, 1), use_bias=not norm))
    if norm:
        model.add(layers.LayerNormalization(epsilon=1e-5))
    model.add(layers.TimeDistributed(layers.Activation(hard_swish)))
    model.add(layers.AvgPool3D((1, tf.math.ceil(in_shape[0] / 32), tf.math.ceil(in_shape[1] / 32)), strides=1))
    model.add(layers.Flatten())
    model.add(layers.Dropout(.075))
    x = model.output
    shared = tfa.layers.NoisyDense(64 if reduced else 192)(x)
    x_v = tfa.layers.NoisyDense(256 if reduced else 224, activation=hard_swish)(x)
    x_v = tfa.layers.NoisyDense(1)(x_v)
    branch1 = branch(x, shared, x_v, out_shape[0], hard_swish)
    branch2 = branch(x, shared, x_v, out_shape[1], hard_swish)
    branch3 = branch(x, shared, x_v, out_shape[2], hard_swish)
    model = Model(model.inputs, [branch1, branch2, branch3])
    # model.summary()
    return model


if __name__ == '__main__':
    m = get_model(in_shape=(256, 256, 3), frames=4, out_shape=(3, 3, 4), reduced=True)
    m.summary()
