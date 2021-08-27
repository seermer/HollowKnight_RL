import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.keras import layers, activations, Sequential, Model, regularizers
from functools import partial


def roundup(num, divisor=8):
    if num % divisor != 0:
        return (num // divisor + 1) * divisor
    else:
        return num


class SE(layers.Layer):
    """
    Squeeze-and-Excitation Networks
    https://arxiv.org/abs/1709.01507
    """

    def __init__(self,
                 in_channels,
                 reduction_ratio,
                 activation1=activations.relu,
                 activation2=activations.hard_sigmoid,
                 dim=2):
        super(SE, self).__init__()
        if dim == 2:
            self.pool = layers.GlobalAvgPool2D()
        elif dim == 3:
            self.pool = layers.GlobalAvgPool3D()
        self.dense1 = layers.Dense(roundup(in_channels // reduction_ratio), activation=activation1)
        self.mult = layers.Multiply()
        self.activation2 = activation2

    def build(self, input_shape):
        self.dense2 = layers.Dense(input_shape[-1], activation=self.activation2)

    def compute_output_shape(self, input_shape):
        return input_shape

    @tf.function
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

    @tf.function
    def call(self, inputs, *args, **kwargs):
        x = self.residual(inputs)
        shortcut = self.shortcut(inputs)
        x = self.add([shortcut, x])
        return self.activation(x)


class ClassicResidual3D(layers.Layer):
    def __init__(self,
                 filters,
                 kernel_size=3,
                 strides=1,
                 activation=partial(tf.nn.leaky_relu, alpha=.1),
                 se=0,
                 stochastic_depth=0.,
                 norm=True,
                 timeless=False,
                 increased=None):
        super(ClassicResidual3D, self).__init__()
        if increased is None:
            increased = 1
        if strides != 1:
            self.shortcut = Sequential([
                layers.AvgPool3D(pool_size=(1, strides, strides), padding="same"),
                layers.Conv3D(filters=filters * increased, kernel_size=1, use_bias=not norm)
            ])
            if norm:
                self.shortcut.add(layers.LayerNormalization(epsilon=1e-5))
        else:
            self.shortcut = lambda x: x

        # stochastic depth: [inputs, residual]
        self.add = layers.Add() if stochastic_depth > 0 else tfa.layers.StochasticDepth(1 - stochastic_depth)

        self.residual = Sequential([

        ])
        if timeless:
            self.residual.add(layers.Conv3D(filters=filters, kernel_size=(1, kernel_size, kernel_size),
                                            strides=(1, strides, strides),
                                            use_bias=not norm, padding="same"))
        else:
            self.residual.add(layers.Conv3D(filters=filters, kernel_size=(kernel_size, 1, 1),
                                            use_bias=not norm, padding="same"))
        if norm:
            self.residual.add(layers.LayerNormalization(epsilon=1e-5))
        self.residual.add(layers.Activation(activation))
        self.residual.add(layers.Conv3D(filters=filters,
                                        kernel_size=(1, kernel_size, kernel_size),
                                        strides=1 if timeless else (1, strides, strides),
                                        use_bias=not norm, padding="same"))
        if norm:
            self.residual.add(layers.LayerNormalization(epsilon=1e-5))
        if increased == 1:
            if se > 0:
                self.residual.add(SE(in_channels=filters, reduction_ratio=se,
                                     activation1=activation, dim=3))
        else:
            self.residual.add(layers.Activation(activation))
            if se > 0:
                self.residual.add(SE(in_channels=filters, reduction_ratio=se,
                                     activation1=activation, dim=3))
            self.residual.add(layers.Conv3D(filters=filters * increased, kernel_size=1,
                                            use_bias=not norm, padding="same"))
            if norm:
                self.residual.add(layers.LayerNormalization(epsilon=1e-5))
        self.strides = strides
        self.filters = filters
        self.norm = norm
        self.increased = increased
        self.activation = activation

    def build(self, input_shape):
        if self.strides == 1 and self.filters * self.increased != input_shape[-1]:
            self.shortcut = Sequential([
                layers.Conv3D(filters=self.filters * self.increased,
                              kernel_size=1, use_bias=not self.norm, padding="same")
            ])
            if self.norm:
                self.shortcut.add(layers.LayerNormalization(epsilon=1e-5))
        del self.norm, self.filters, self.strides

    def compute_output_shape(self, input_shape):
        return self.residual.compute_output_shape(input_shape)

    @tf.function
    def call(self, inputs, *args, **kwargs):
        x = self.residual(inputs)
        shortcut = self.shortcut(inputs)
        x = self.add([shortcut, x])
        return self.activation(x)


class SpaceToDepthStem(layers.Layer):
    def __init__(self, block_size=4, filters=64, norm=True, activation=tf.nn.leaky_relu):
        super(SpaceToDepthStem, self).__init__()
        self.func = partial(tf.nn.space_to_depth, block_size=block_size)
        self.conv = Sequential([layers.Conv2D(filters, 3, padding="same", use_bias=not norm)])
        if norm:
            self.conv.add(layers.LayerNormalization())
        self.conv.add(layers.Activation(activation))
        self.block_size = block_size
        self.filters = filters

    def compute_output_shape(self, input_shape):
        return (None, input_shape[1] // self.block_size, input_shape[2] // self.block_size,
                input_shape[-1] * (self.block_size ** 2))

    @tf.function
    def call(self, inputs, *args, **kwargs):
        x = self.func(inputs)
        return self.conv(x)


def grad_rescale_outer(n):
    @tf.custom_gradient
    def grad_rescale(x):
        def custom_grad(dy):
            return tf.divide(dy, n)

        return x, custom_grad

    return grad_rescale


class GradientRescale(layers.Layer):
    def __init__(self, n):
        super(GradientRescale, self).__init__()
        self.rescale_func = grad_rescale_outer(n)

    @tf.function
    def call(self, inputs, *args, **kwargs):
        return self.rescale_func(inputs)


def branch(x, n, activation):
    x = tfa.layers.NoisyDense(128, activation=activation)(x)
    x = GradientRescale(3.)(x)
    x = tfa.layers.NoisyDense(n)(x)
    return x


def get_main_model(in_shape, frames, out_shape, norm=True, no_se=False):
    model = Sequential()
    model.add(SpaceToDepthStem(filters=64, norm=norm))
    model.add(ClassicResidual(64, 3, strides=1, se=0 if no_se else 4, norm=norm))
    model.add(ClassicResidual(64, 3, strides=1, norm=norm))
    model.add(ClassicResidual(96, 3, strides=2, se=0 if no_se else 4, norm=norm))
    model = Sequential([layers.InputLayer((frames,) + in_shape),
                        layers.TimeDistributed(model)])
    model.add(ClassicResidual3D(96, 3, strides=1, se=0 if no_se else 6, norm=norm, timeless=True))
    model.add(ClassicResidual3D(112, 3, strides=2, se=0 if no_se else 8, norm=norm, increased=4))
    model.add(ClassicResidual3D(112, 3, strides=1, se=0 if no_se else 8, norm=norm, increased=4))
    model.add(ClassicResidual3D(160, 3, strides=2, norm=norm, increased=4))
    model.add(ClassicResidual3D(160, 3, strides=1, norm=norm, increased=4))
    model.add(layers.AvgPool3D((frames, tf.math.ceil(in_shape[0] / 32), tf.math.ceil(in_shape[1] / 32)),
                               strides=1))
    model.add(layers.Flatten())
    model.add(GradientRescale(tf.math.sqrt(2.)))
    out = model.output
    branch1 = branch(out, out_shape[0], tf.nn.leaky_relu)
    branch2 = branch(out, out_shape[1], tf.nn.leaky_relu)
    branch3 = branch(out, out_shape[2], tf.nn.leaky_relu)
    model = Model(model.inputs, [branch1, branch2, branch3])
    return model, out


def get_model(in_shape, frames, out_shape, norm=True, no_se=False):
    main_model, branching = get_main_model(in_shape, frames, out_shape, norm, no_se)
    v = layers.Dense(128, activation=tf.nn.leaky_relu)(branching)
    v = layers.Dense(1)(v)
    branches = []
    for a in main_model.outputs:
        branches.append(layers.Lambda(lambda val:
                                      tf.add(tf.add(val[0], -tf.reduce_mean(val[0], axis=-1, keepdims=True)),
                                             val[1]))([a, v]))
    final = Model(main_model.inputs, branches)
    del main_model
    return final


if __name__ == '__main__':
    f = get_model(in_shape=(256, 224, 3), frames=4, out_shape=(3, 3, 4), norm=True)
    # m.summary()
    f.summary()
    # print(tf.config.experimental.get_memory_info("GPU:0"))
    # f.compile(loss=["mse" for _ in range(3)])
    # print(f(tf.random.uniform((2, 4, 256, 224, 3))))
    # f.fit(tf.random.uniform((2, 4, 256, 224, 3)),
    #       [tf.convert_to_tensor([[1., 1., 2.], [1., 1., 2.]]), tf.convert_to_tensor([[2., 1., 1.], [1., 1., 2.]]),
    #        tf.convert_to_tensor([[1., 2., 3., 4.], [2., 1., 1., 2.]])], epochs=3)
