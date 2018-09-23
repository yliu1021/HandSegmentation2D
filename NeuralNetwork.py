import tensorflow as tf
from NetworkLayers import NetworkLayers as Net


class Network:

    def __init__(self, sess, layers, trainable=True):
        input_shape = [72, 128, 3]
        self.sess = sess
        with sess.graph.as_default():
            self.input_tf = tf.placeholder(tf.float32, shape=[None] + input_shape, name="input_tf")
            prev_layer = self.input_tf
            for i, layer in enumerate(layers, 1):
                prev_layer = layer(prev_layer, "layer_%d" % i, trainable)
            output_shape = prev_layer.get_shape().as_list()
            if output_shape[:2] != [72, 128]:
                prev_layer = tf.image.resize_images(prev_layer, [72, 128])
            prev_layer = Net.conv(prev_layer, "penultimate_projection",
                                  kernel_size=3, stride=1, out_chan=1,
                                  add_bias=False, trainable=trainable)
            self.pred_tf = tf.identity(prev_layer, name="pred_tf")

    def restore(self, path):
        with self.sess.graph.as_default():
            saver = tf.train.Saver()
            saver.restore(self.sess, path)

    def initialize(self):
        with self.sess.graph.as_default():
            self.sess.run(tf.global_variables_initializer())
