# coding=utf-8
import tensorflow as tf
from data_helper import Dataset
from math import sqrt
from config import config

class CharCNN(object):
    """
    A CNN for text classification
    """
    def __init__(self, l0, num_classes, conv_layers, fc_layers, l2_reg_lambda=0.0):
        # Placeholders for input, output and dropout
        self.input_x = tf.placeholder(tf.int32, [None, l0], name='input_x')
        self.input_y = tf.placeholder(tf.float32, [None, num_classes], name='input_y')
        self.dropout_keep_prob = tf.placeholder(tf.float32, name='dropout_keep_prob')

        # Keeping track of l2 regularization loss (optional)
        l2_loss = tf.constant(0.0)

        # Embedding layer
        with tf.device('/cpu:0'), tf.name_scope('embedding'):
            train_data = Dataset(config.train_data_source)
            self.W, _ = train_data.onehot_dic_build()
            self.x_image = tf.nn.embedding_lookup(self.W, self.input_x)
            self.x_flat = tf.expand_dims(self.x_image, axis=-1)
            print(self.x_flat.shape) # [None, l0, 69, 1]

        for i, layer_configs in enumerate(conv_layers):
            num_filters = layer_configs[0]
            filter_size = layer_configs[1]
            pool_size = layer_configs[2]
            width = self.x_flat.get_shape()[2].value

            with tf.name_scope("conv_layer-%s" % (i+1)):
                W = tf.Variable(initial_value=tf.truncated_normal(shape=[filter_size, width, 1, num_filters], stddev=0.02),
                                name='W')
                b = tf.Variable(tf.zeros(shape=[num_filters], name='b'))
                conv = tf.nn.conv2d(self.x_flat, W, strides=[1, 1, 1, 1], padding='VALID')
                conv = tf.nn.bias_add(conv, b)
                if pool_size != None:
                    pool = tf.nn.max_pool(conv, [1, pool_size, 1, 1], strides=[1, pool_size, 1, 1], padding='VALID')
                else:
                    pool = conv

                self.x_flat = tf.transpose(pool, [0, 1, 3, 2], name='transpose')

        # print(self.x_flat.shape) # [None, 34, 256, 1]

        with tf.name_scope('reshape'):
            fc_dim = self.x_flat.get_shape()[1].value * self.x_flat.get_shape()[2].value
            self.x_flat = tf.reshape(self.x_flat, shape=[-1, fc_dim]) # [None, 8704]

        for i, units in enumerate(fc_layers):
            with tf.name_scope('fc_layer-%s' %(i + 1)):
                W = tf.Variable(initial_value=tf.truncated_normal(shape=[self.x_flat.get_shape()[1].value, units], stddev=0.05))
                b = tf.Variable(initial_value=tf.zeros(shape=[units]))
                self.x_flat = tf.nn.relu(tf.matmul(self.x_flat, W) + b)

                with tf.name_scope('dropout'):
                    self.x_flat = tf.nn.dropout(self.x_flat, self.dropout_keep_prob)

        with tf.name_scope('output_layer'):
            W = tf.Variable(tf.random_normal(shape=[fc_layers[-1], num_classes]), name='W')
            b = tf.Variable(tf.zeros(shape=[num_classes]), name='b')
            self.y_pred = tf.nn.xw_plus_b(self.x_flat, W, b, name='y_pred')
            self.predictions = tf.argmax(self.y_pred, 1, name='predictions')

        # Calculate Mean Cross-entropy
        with tf.name_scope('loss'):
            losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.y_pred, labels=self.input_y)
            self.loss = tf.reduce_mean(losses) + l2_reg_lambda * l2_loss

        # Accuracy
        with tf.name_scope('accuracy'):
            correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, 'float'), name='accuracy')




# cnn = CharCNN(l0=1014, num_classes=4, conv_layers=config.model.conv_layers, fc_layers=[2048, 2048])



