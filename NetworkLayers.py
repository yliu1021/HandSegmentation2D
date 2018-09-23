import tensorflow as tf
import numpy as np
import math


class NetworkLayers:
    """ Operations that are frequently used within networks. """
    neg_slope_of_relu = 0.01

    def __init__(self):
        pass

    @classmethod
    def inception_module_simple(cls, in_tensor, intermediate_chan, layer_name, trainable=True):
        with tf.variable_scope(layer_name):
            conv_1x1_3x3_factor = cls.conv(in_tensor, layer_name="1x1of3x3factor",
                                           kernel_size=1, stride=1, out_chan=intermediate_chan,
                                           trainable=trainable, add_bias=False)
            conv_1x1_3x1_factor = cls.conv(conv_1x1_3x3_factor, layer_name="3x1of3x3factor",
                                           kernel_size=[3, 1], stride=1, out_chan=intermediate_chan,
                                           trainable=trainable, add_bias=False)
            conv_1x1_1x3_factor = cls.conv(conv_1x1_3x3_factor, layer_name="1x3of3x3factor",
                                           kernel_size=[1, 3], stride=1, out_chan=intermediate_chan,
                                           trainable=trainable, add_bias=False)

            conv_1x1 = cls.conv(in_tensor, layer_name="1x1factor",
                                kernel_size=1, stride=1, out_chan=intermediate_chan,
                                trainable=trainable, add_bias=False)

            return tf.concat([conv_1x1_3x1_factor, conv_1x1_1x3_factor,
                              conv_1x1], axis=3, name="out")

    @classmethod
    def inception_module_base(cls, in_tensor, intermediate_chan, layer_name, trainable=True):
        with tf.variable_scope(layer_name):
            conv_5x5_1x1_factor = cls.conv(in_tensor, layer_name="1x1of5x5factor",
                                           kernel_size=1, stride=1, out_chan=intermediate_chan,
                                           trainable=trainable, add_bias=False)
            conv_5x5_3x3_factor = cls.conv(conv_5x5_1x1_factor, layer_name="3x3of5x5factor",
                                           kernel_size=3, stride=1, out_chan=intermediate_chan*2,
                                           trainable=trainable, add_bias=False)
            conv_5x5_3x1_factor = cls.conv(conv_5x5_3x3_factor, layer_name="3x1of5x5factor",
                                           kernel_size=[3, 1], stride=1, out_chan=intermediate_chan*2,
                                           trainable=trainable, add_bias=False)
            conv_5x5_1x3_factor = cls.conv(conv_5x5_3x3_factor, layer_name="1x3of5x5factor",
                                           kernel_size=[1, 3], stride=1, out_chan=intermediate_chan*2,
                                           trainable=trainable, add_bias=False)

            conv_3x3_1x1_factor = cls.conv(in_tensor, layer_name="1x1of3x3factor",
                                           kernel_size=1, stride=1, out_chan=intermediate_chan,
                                           trainable=trainable, add_bias=False)
            conv_3x3_3x1_factor = cls.conv(conv_3x3_1x1_factor, layer_name="3x1of3x3factor",
                                           kernel_size=[3, 1], stride=1, out_chan=intermediate_chan*2,
                                           trainable=trainable, add_bias=False)
            conv_3x3_1x3_factor = cls.conv(conv_3x3_1x1_factor, layer_name="1x3of3x3factor",
                                           kernel_size=[1, 3], stride=1, out_chan=intermediate_chan*2,
                                           trainable=trainable, add_bias=False)

            conv_1x1 = cls.conv(in_tensor, layer_name="1x1factor",
                                kernel_size=1, stride=1, out_chan=intermediate_chan,
                                trainable=trainable, add_bias=False)

            conv_pool = tf.nn.max_pool(in_tensor, ksize=[1, 3, 3, 1], strides=[1, 1, 1, 1],
                                       padding="SAME", name="max_pool")
            conv_1x1_pool = cls.conv(conv_pool, layer_name="max_pool_conv",
                                     kernel_size=1, stride=1, out_chan=intermediate_chan*2,
                                     trainable=trainable, add_bias=False)

            return tf.concat([conv_5x5_3x1_factor, conv_5x5_1x3_factor,
                              conv_3x3_3x1_factor, conv_3x3_1x3_factor,
                              conv_1x1, conv_1x1_pool], axis=3, name="out")

    @classmethod
    def mobilenetv1_module_base(cls, in_tensor, layer_name, expansion_factor, out_chan, trainable=True):
        """The base mobilenet block without the residual layer"""
        with tf.variable_scope(layer_name):
            in_size = in_tensor.get_shape().as_list()
            expanded_size = in_size[3] * expansion_factor

            expansion_conv = cls.conv(in_tensor, "expansion_conv", 1, 1,
                                      expanded_size, trainable=trainable)
            expansion_relu = cls.leaky_relu(expansion_conv, cap=6, name="expansion_relu6")

            depthwise_conv = cls.separable_conv(expansion_relu, "depthwise_conv", 1, 1,
                                                expanded_size, trainable=trainable)
            depthwise_relu = cls.leaky_relu(depthwise_conv, cap=6, name="depthwise_relu6")

            projection_conv = cls.conv(depthwise_relu, "projection_conv", 1, 1, out_chan, trainable=trainable)

            out_tensor = tf.identity(projection_conv, "out")
            return out_tensor

    @classmethod
    def mobilenetv1_module_residual(cls, in_tensor, layer_name, expansion_factor, out_chan,
                                    trainable=True):
        """Includes final residual connection"""
        with tf.variable_scope(layer_name):
            base_block = cls.mobilenetv1_module_base(in_tensor, "base_block", expansion_factor, out_chan,
                                                     trainable=trainable)
            out_tensor = tf.identity(base_block, name="out")
            if in_tensor.get_shape()[3] == out_chan:
                out_tensor = tf.add(in_tensor, base_block, name="residual_add")
            return out_tensor

    @classmethod
    def batch_norm(cls, in_tensor, layer_name, trainable=True, training=False):
        with tf.variable_scope(layer_name):
            out_tensor = tf.layers.batch_normalization(in_tensor, name="batch_norm", fused=True,
                                                       trainable=trainable, training=training)
            return out_tensor

    @classmethod
    def leaky_relu(cls, tensor, cap=None, name='relu'):
        out_tensor = tf.maximum(tensor, cls.neg_slope_of_relu * tensor, name=name)
        if cap is not None:
            out_tensor = tf.minimum(out_tensor, cls.neg_slope_of_relu * (out_tensor - cap) + cap, name=name+"mincap")
        return out_tensor

    @classmethod
    def separable_conv(cls, in_tensor, layer_name, kernel_size, stride, out_chan, trainable=True):
        with tf.variable_scope(layer_name):
            in_size = in_tensor.get_shape().as_list()

            strides = [1, stride, stride, 1]
            depthwise_kernel_shape = None
            if kernel_size is list:
                if len(kernel_size) == 2:
                    depthwise_kernel_shape = kernel_size.extend([in_size[3], 1])
                elif len(kernel_size) == 3:
                    depthwise_kernel_shape = kernel_size.append(1)
                elif len(kernel_size) == 4:
                    depthwise_kernel_shape = kernel_size
            else:
                depthwise_kernel_shape = [kernel_size, kernel_size, in_size[3], 1]

            pointwise_kernel_shape = [1, 1, depthwise_kernel_shape[-1]*depthwise_kernel_shape[-2], out_chan]

            depthwise_kernel = tf.get_variable('depthwise', depthwise_kernel_shape, tf.float32,
                                               tf.contrib.layers.xavier_initializer(), trainable=trainable,
                                               collections=['wd', 'variables', 'filters'])
            pointwise_kernel = tf.get_variable('pointwise', pointwise_kernel_shape, tf.float32,
                                               tf.contrib.layers.xavier_initializer(), trainable=trainable,
                                               collections=['wd', 'variables', 'filters'])
            out_tensor = tf.nn.separable_conv2d(in_tensor, depthwise_kernel, pointwise_kernel, strides,
                                                padding='SAME', name='out')

            return out_tensor

    @classmethod
    def relu(cls, tensor, name='relu'):
        return cls.leaky_relu(tensor, name=name)

    @classmethod
    def conv(cls, in_tensor, layer_name, kernel_size, stride, out_chan, add_bias=True, trainable=True):
        with tf.variable_scope(layer_name):
            in_size = in_tensor.get_shape().as_list()

            strides = [1, stride, stride, 1]
            kernel_shape = []
            if type(kernel_size) is list:
                if len(kernel_size) == 2:
                    kernel_shape = kernel_size + [in_size[3], out_chan]
                elif len(kernel_size) == 3:
                    kernel_shape = kernel_size + [out_chan]
                elif len(kernel_size) == 4:
                    kernel_shape = kernel_size
            else:
                kernel_shape = [kernel_size, kernel_size, in_size[3], out_chan]

            kernel = tf.get_variable('weights', kernel_shape, tf.float32,
                                     tf.contrib.layers.xavier_initializer_conv2d(), trainable=trainable,
                                     collections=['wd', 'variables', 'filters'])
            tmp_result = tf.nn.conv2d(in_tensor, kernel, strides, padding='SAME')

            if add_bias:
                biases = tf.get_variable('biases', [kernel_shape[3]], tf.float32,
                                         tf.constant_initializer(0.0001), trainable=trainable,
                                         collections=['wd', 'variables', 'biases'])
                out_tensor = tf.nn.bias_add(tmp_result, biases, name='out')
            else:
                out_tensor = tmp_result

            return out_tensor

    @classmethod
    def conv_relu(cls, in_tensor, layer_name, kernel_size, stride, out_chan, trainable=True):
        tensor = cls.conv(in_tensor, layer_name, kernel_size, stride, out_chan, trainable)
        out_tensor = cls.leaky_relu(tensor, name='out')
        return out_tensor

    @classmethod
    def resize_images(cls, in_tensor, shape):
        out = tf.image.resize_images(in_tensor, shape)
        return out

    @classmethod
    def avg_pool(cls, bottom, name='avg_pool'):
        pooled = tf.nn.avg_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                                padding='SAME', name=name)
        return pooled

    @classmethod
    def max_pool(cls, bottom, name='pool'):
        pooled = tf.nn.max_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                                padding='VALID', name=name)
        return pooled

    @classmethod
    def upconv(cls, in_tensor, layer_name, filters, kernel_size, stride, trainable=True):
        with tf.variable_scope(layer_name):
            kernel_shape = [kernel_size, kernel_size]
            strides = [stride, stride]

            out_tensor = tf.layers.conv2d_transpose(inputs=in_tensor, filters=filters, kernel_size=kernel_shape,
                                                    strides=strides, name="deconv", trainable=trainable)
            return out_tensor

    @classmethod
    def _upconv(cls, in_tensor, layer_name, output_shape, kernel_size, stride, trainable=True):
        with tf.variable_scope(layer_name):
            in_size = in_tensor.get_shape().as_list()

            kernel_shape = [kernel_size, kernel_size, in_size[3], in_size[3]]
            strides = [1, stride, stride, 1]

            kernel = cls.get_deconv_filter(kernel_shape, trainable)
            tmp_result = tf.nn.conv2d_transpose(value=in_tensor, filter=kernel, output_shape=output_shape,
                                                strides=strides, padding='SAME')

            biases = tf.get_variable('biases', [kernel_shape[2]], tf.float32,
                                     tf.constant_initializer(0.0), trainable=trainable,
                                     collections=['wd', 'variables', 'biases'])
            out_tensor = tf.nn.bias_add(tmp_result, biases)
            return out_tensor

    @classmethod
    def _upconv_relu(cls, in_tensor, layer_name, output_shape, kernel_size, stride, trainable=True):
        tensor = cls.upconv(in_tensor, layer_name, output_shape, kernel_size, stride, trainable)
        out_tensor = cls.leaky_relu(tensor, name='out')
        return out_tensor

    @staticmethod
    def get_deconv_filter(f_shape, trainable):
        width = f_shape[0]
        height = f_shape[1]
        f = math.ceil(width/2.0)
        c = (2 * f - 1 - f % 2) / (2.0 * f)
        bilinear = np.zeros([f_shape[0], f_shape[1]])
        for x in range(width):
            for y in range(height):
                value = (1 - abs(x / f - c)) * (1 - abs(y / f - c))
                bilinear[x, y] = value
        weights = np.zeros(f_shape)
        for i in range(f_shape[2]):
            weights[:, :, i, i] = bilinear

        init = tf.constant_initializer(value=weights,
                                       dtype=tf.float32)
        return tf.get_variable(name="weights", initializer=init,
                               shape=weights.shape, trainable=trainable, collections=['wd', 'variables', 'filters'])

    @staticmethod
    def fully_connected(in_tensor, layer_name, out_chan, trainable=True):
        with tf.variable_scope(layer_name):
            in_size = in_tensor.get_shape().as_list()
            assert len(in_size) == 2, 'Input to a fully connected layer must be a vector.'
            weights_shape = [in_size[1], out_chan]

            # weight matrix
            weights = tf.get_variable('weights', weights_shape, tf.float32,
                                      tf.contrib.layers.xavier_initializer(), trainable=trainable)
            weights = tf.check_numerics(weights, 'weights: %s' % layer_name)

            # bias
            biases = tf.get_variable('biases', [out_chan], tf.float32,
                                     tf.constant_initializer(0.0001), trainable=trainable)
            biases = tf.check_numerics(biases, 'biases: %s' % layer_name)

            out_tensor = tf.matmul(in_tensor, weights) + biases
            return out_tensor

    @classmethod
    def fully_connected_relu(cls, in_tensor, layer_name, out_chan, trainable=True):
        tensor = cls.fully_connected(in_tensor, layer_name, out_chan, trainable)
        out_tensor = tf.maximum(tensor, cls.neg_slope_of_relu*tensor, name='out')
        return out_tensor

    @staticmethod
    def dropout(in_tensor, keep_prob):
        """ Dropout: Each neuron is dropped independently. """
        with tf.variable_scope('dropout'):
            tensor_shape = in_tensor.get_shape().as_list()
            out_tensor = tf.nn.dropout(in_tensor, keep_prob, noise_shape=tensor_shape)
            return out_tensor

    @staticmethod
    def spatial_dropout(in_tensor, keep_prob, evaluation):
        """ Spatial dropout: Not each neuron is dropped independently, but feature map wise. """
        with tf.variable_scope('spatial_dropout'):
            tensor_shape = in_tensor.get_shape().as_list()
            out_tensor = tf.cond(evaluation,
                                 lambda: tf.nn.dropout(in_tensor, 1.0,
                                                       noise_shape=tensor_shape),
                                 lambda: tf.nn.dropout(in_tensor, keep_prob,
                                                       noise_shape=[tensor_shape[0], 1, 1, tensor_shape[3]]))
            return out_tensor
