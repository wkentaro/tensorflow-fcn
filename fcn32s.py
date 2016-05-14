from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import math

import numpy as np
import tensorflow as tf

from fcn import pascal


def softmax_cross_entropy(x, label):
    shape = tf.convert_to_tensor([1, -1], dtype=tf.int32, name='shape')
    x = tf.reshape(x, shape)
    label = tf.reshape(label, shape)
    y = tf.nn.softmax(x)
    loss = tf.gather_nd(y, tf.tile(label, [1, 21]))
    mean_loss = - tf.div(tf.reduce_sum(tf.log(loss), keep_dims=True),
                         tf.to_float(tf.size(label)))
    return mean_loss


class FCN32s(object):
    def __init__(self, vgg16_npy_path):
        self.data_dict = np.load(vgg16_npy_path, encoding='latin1').item()
        print("npy file loaded")

    def build(self, train=False):
        x = tf.placeholder(tf.float32, [None, None, None, 3], name='x')
        if train:
            height, width = x.get_shape().as_list()[1:3]
            label = tf.placeholder(
                tf.int32, [None, height, width, 1], name='label')

        self.conv1_1 = self._conv_layer(
            x, 'conv1_1', n_output=64, ksize=3, padding='VALID',
            use_pretrained=True)
        self.conv1_2 = self._conv_layer(
            self.conv1_1, 'conv1_2', n_output=64, ksize=3, padding='VALID',
            use_pretrained=True)
        self.pool1 = self._max_pool(
            self.conv1_2, 'pool1', ksize=2, strides=[1, 2, 2, 1])  # 1/2

        self.conv2_1 = self._conv_layer(
            self.pool1, 'conv2_1', n_output=128, ksize=3, padding='VALID',
            use_pretrained=True)
        self.conv2_2 = self._conv_layer(
            self.conv2_1, 'conv2_2', n_output=128, ksize=3, padding='VALID',
            use_pretrained=True)
        self.pool2 = self._max_pool(
            self.conv2_2, 'pool2', ksize=2, strides=[1, 2, 2, 1])  # 1/4

        self.conv3_1 = self._conv_layer(
            self.pool2, 'conv3_1', n_output=256, ksize=3, padding='VALID',
            use_pretrained=True)
        self.conv3_2 = self._conv_layer(
            self.conv3_1, 'conv3_2', n_output=256, ksize=3, padding='VALID',
            use_pretrained=True)
        self.conv3_2 = self._conv_layer(
            self.conv3_2, 'conv3_3', n_output=256, ksize=3, padding='VALID',
            use_pretrained=True)
        self.pool3 = self._max_pool(
            self.conv3_2, 'pool3', ksize=2, strides=[1, 2, 2, 1])  # 1/8

        self.conv4_1 = self._conv_layer(
            self.pool3, 'conv4_1', n_output=512, ksize=3, padding='VALID',
            use_pretrained=True)
        self.conv4_2 = self._conv_layer(
            self.conv4_1, 'conv4_2', n_output=512, ksize=3, padding='VALID',
            use_pretrained=True)
        self.conv4_3 = self._conv_layer(
            self.conv4_2, 'conv4_3', n_output=512, ksize=3, padding='VALID',
            use_pretrained=True)
        self.pool4 = self._max_pool(
            self.conv4_3, 'pool4', ksize=2, strides=[1, 2, 2, 1])  # 1/16

        self.conv5_1 = self._conv_layer(
            self.pool4, 'conv5_1', n_output=512, ksize=3, padding='VALID',
            use_pretrained=True)
        self.conv5_2 = self._conv_layer(
            self.conv5_1, 'conv5_2', n_output=512, ksize=3, padding='VALID',
            use_pretrained=True)
        self.conv5_3 = self._conv_layer(
            self.conv5_2, 'conv5_3', n_output=512, ksize=3, padding='VALID',
            use_pretrained=True)
        self.pool5 = self._max_pool(
            self.conv5_3, 'pool5', ksize=2, strides=[1, 2, 2, 1])  # 1/32

        self.fc6 = self._conv_layer(self.pool5, 'fc6', n_output=4096, ksize=7)
        self.relu6 = tf.nn.relu(self.fc6)  # 1/32
        if train:
            self.relu6 = tf.nn.dropout(self.relu6, 0.5)

        self.fc7 = self._conv_layer(self.relu6, 'fc7', n_output=4096, ksize=1)
        self.relu7 = tf.nn.relu(self.fc7)  # 1/32
        if train:
            self.relu7 = tf.nn.dropout(self.relu7, 0.5)

        self.score_fr = self._conv_layer(
            self.relu7, 'score_fr', n_output=21, ksize=1)  # 1/32

        output_shape = x.get_shape().as_list()
        output_shape[3] = 21
        output_shape = tf.placeholder(tf.int32, output_shape, name='output_shape')
        self.upscore = self._deconv_layer(
            self.score_fr, 'upscore', ksize=64, output_shape=output_shape,
            strides=[1, 32, 32, 1])  # 1

        if train:
            loss = softmax_cross_entropy(self.upscore, label)
            return {'x': x, 'label': label, 'loss': loss}

    def _max_pool(self, bottom, name, ksize=None, strides=None, padding=None):
        if ksize is None:
            ksize_h = ksize_w = 1
        elif isinstance(ksize, int):
            ksize_h = ksize_w = ksize
        else:
            assert isinstance(ksize, (list, tuple))
            assert len(ksize) == 2
            ksize_h, ksize_w = ksize
        ksize = [1, ksize_h, ksize_w, 1]
        strides = strides or [1, 1, 1, 1]
        padding = padding or 'SAME'
        return tf.nn.max_pool(bottom, ksize, strides, padding, name=name)

    def _conv_layer(self, bottom, name, n_output, ksize,
                    strides=None, padding=None, use_pretrained=False):
        strides = strides or [1, 1, 1, 1]
        padding = padding or 'SAME'
        if isinstance(ksize, int):
            ksize_h = ksize_w = ksize
        else:
            assert isinstance(ksize, (list, tuple))
            assert len(ksize) == 2
            ksize_h, ksize_w = ksize
        # n_input: (0: batch, 1: height, 2: width, 3: channel)
        n_input = bottom.get_shape().as_list()[3]
        with tf.variable_scope(name):
            if use_pretrained:
                filt = self.get_conv_filter(name)
                conv_biases = self.get_bias(name)
                assert filt.get_shape().as_list()[3] == n_output
                assert conv_biases.get_shape().as_list()[0] == n_output
            else:
                filt = tf.Variable(
                    tf.random_uniform(
                        [ksize_h, ksize_w, n_input, n_output],
                        -1 / math.sqrt(n_input), 1 / math.sqrt(n_input)))
                conv_biases = tf.Variable(tf.zeros([n_output]))
            conv = tf.nn.conv2d(bottom, filt, strides, padding)
            bias = tf.nn.bias_add(conv, conv_biases)
            relu = tf.nn.relu(bias)
            return relu

    def _deconv_layer(self, bottom, name, ksize, output_shape,
                      strides=None, padding=None):
        strides = strides or [1, 1, 1, 1]
        padding = padding or 'SAME'
        if isinstance(ksize, int):
            ksize_h = ksize_w = ksize
        else:
            assert isinstance(ksize, (list, tuple))
            assert len(ksize) == 2
            ksize_h, ksize_w = ksize
        # (0: batch, 1: height, 2: width, 3: channel)
        n_input = bottom.get_shape().as_list()[3]
        n_output = output_shape.get_shape().as_list()[3]
        with tf.variable_scope(name):
            filt = tf.Variable(
                tf.random_uniform(
                    [ksize_h, ksize_w, n_input, n_output],
                    -1 / math.sqrt(n_input), 1 / math.sqrt(n_input)))
            deconv = tf.nn.conv2d_transpose(
                bottom, filt, output_shape, strides, padding)
            deconv_biases = tf.Variable(tf.zeros([n_output]))
            bias = tf.nn.bias_add(deconv, deconv_biases)
            relu = tf.nn.relu(bias)
            return relu

    def get_conv_filter(self, name):
        return tf.Variable(self.data_dict[name][0], name='filter')

    def get_bias(self, name):
        return tf.Variable(self.data_dict[name][1], name='biases')


def main():
    x = tf.placeholder(tf.float32, [None, None, None, 3], name='x')

    fcn = FCN32s('vgg16.npy').build(train=True)

    learning_rate = 1e-10
    momentum = 0.99
    optimizer = tf.train.MomentumOptimizer(learning_rate, momentum)

    sess = tf.Session()
    sess.run(tf.initialize_all_variables())

    dataset = pascal.SegmentationClassDataset()

    max_iteration = 100000
    for i_iter in xrange(max_iteration):
        datum = dataset.next_batch(batch_size=1, type='train')
        rgb = (datum.img - np.array([103.939, 116.779, 123.68]))
        bgr = rgb[::-1]
        x = bgr[np.newaxis, ...]
        label = datum.label[np.newaxis, ...]
        sess.run(optimizer, feed_dict={fcn['x']: x, fcn['label']: label})
        print(i_iter, sess.run(fcn['loss'],
                               feed_dict={fcn['x']: x, fcn['label']: label}))


if __name__ == '__main__':
    main()
