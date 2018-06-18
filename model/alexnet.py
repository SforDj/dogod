import tensorflow as tf
import numpy as np
from data_preprocess.constant import *
from data_preprocess.data_integrated import Dog
from layers.cnn_utils import *
import datetime


class Config:
    batch_size = 1
    learning_rate = 0.00001
    keep_prob = 0.5


class Alexnet:
    def __init__(self, config):
        self.batch_size = config.batch_size
        self.learning_rate = config.learning_rate
        self.n_train_samples = 8153
        self.keep_prob = config.keep_prob
        self.dogs = Dog(base_dir)
        self.n_class = self.dogs.n_class

    def alexnet_graph(self, inputs):
        conv1 = conv_2d(inputs, name="conv1", w_shape=[11, 11, 3, 96], b_shape=[96],
                        strides=[1, 4, 4, 1], padding="VALID", active_func=tf.nn.relu)
        pool1 = max_pool_2d(conv1, name="pool1", ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding="VALID")
        lrn1 = lrn_layer(pool1, depth_radius=2, alpha=2e-05, beta=0.75, name="lrn1")

        conv2 = conv_2d(lrn1, name="conv2", w_shape=[5, 5, 96, 256], b_shape=[256],
                        strides=[1, 1, 1, 1], padding="SAME", active_func=tf.nn.relu, groups=2)
        pool2 = max_pool_2d(conv2, name="pool2", ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding="VALID")
        lrn2 = lrn_layer(pool2, depth_radius=2, alpha=2e-05, beta=0.75, name="lrn2")

        conv3 = conv_2d(lrn2, name="conv3", w_shape=[3, 3, 256, 384], b_shape=[384],
                        strides=[1, 1, 1, 1], padding="SAME", active_func=tf.nn.relu)
        conv4 = conv_2d(conv3, name="conv4", w_shape=[3, 3, 384, 384], b_shape=[384],
                        strides=[1, 1, 1, 1], padding="SAME", active_func=tf.nn.relu, groups=1)
        conv5 = conv_2d(conv4, name="conv5", w_shape=[3, 3, 384, 256], b_shape=[256],
                        strides=[1, 1, 1, 1], padding="SAME", active_func=tf.nn.relu, groups=1)
        pool5 = max_pool_2d(conv5, name="pool5", ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding="VALID")

        pool5_shape = pool5.get_shape().as_list()
        pool5_reshaped_nodes = pool5_shape[1] * pool5_shape[2] * pool5_shape[3]
        pool5_reshaped = tf.reshape(pool5, shape=[-1, pool5_reshaped_nodes])

        fc6 = fully_connected(pool5_reshaped, name="fc6", w_shape=[pool5_reshaped_nodes, 4096], b_shape=[4096],
                              need_dropout=True, keep_prob=self.keep_prob)

        fc7 = fully_connected(fc6, name="fc7", w_shape=[4096, 4096], b_shape=[4096],
                              need_dropout=True, keep_prob=self.keep_prob)

        fc8 = fully_connected(fc7, name="tc8", w_shape=[4096, self.n_class], b_shape=[self.n_class],
                              need_dropout=False, active_func=None)
        return fc8

    def loss_graph(self, logits, targets):
        y_one_hot = tf.one_hot(targets, self.n_class)
        y_reshaped = tf.reshape(y_one_hot, shape=[-1, self.n_class])
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y_reshaped))
        acc = tf.reduce_mean(tf.cast(tf.equal(tf.cast(tf.argmax(logits, axis=-1), tf.int32),
                                                   tf.cast(tf.argmax(y_reshaped, axis=-1), tf.int32)), tf.float32))
        return loss, acc

    def optimizer_graph(self, loss):
        optimizer = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(loss)
        return optimizer

    def train(self, epoch):
        inputs = tf.placeholder(tf.float32, (self.batch_size, 227, 227, 3))
        targets = tf.placeholder(tf.int32, (self.batch_size, 1))

        logits = self.alexnet_graph(inputs)
        loss, acc = self.loss_graph(logits, targets)
        optimizer = self.optimizer_graph(loss)

        d = self.dogs
        train_data_raw, train_label_raw = d.train_data, d.train_label
        # batch_train_data, batch_train_label = tf.train.batch([train_data_raw, train_label_raw],
        #                                                      batch_size=self.batch_size,
        #                                                      capacity=self.n_train_samples,
        #                                                      num_threads=1)

        batch_train_data, batch_train_label = tf.train.shuffle_batch([train_data_raw, train_label_raw],
                                                                     batch_size=self.batch_size,
                                                                     capacity=self.n_train_samples,
                                                                     min_after_dequeue=1,
                                                                     num_threads=1)
            # tf.train.batch([train_data_raw, train_label_raw],
            #                                                  batch_size=self.batch_size,
            #                                                  capacity=self.n_train_samples,
            #                                                  num_threads=1)


        valid_data_raw, valid_label_raw = d.valid_data, d.valid_label
        batch_valid_data, batch_valid_label = tf.train.batch([valid_data_raw, valid_label_raw],
                                                             batch_size=self.batch_size,
                                                             capacity=self.n_train_samples,
                                                             num_threads=1)


        saver = tf.train.Saver()
        sess_config = tf.ConfigProto(allow_soft_placement=True)
        sess_config.gpu_options.allow_growth = True

        batch_num = self.n_train_samples // self.batch_size + 1

        with tf.Session(config=sess_config) as sess:
            sess.run(tf.global_variables_initializer())
            step = 0
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)
            for i in range(epoch):
                epoch_loss = 0
                epoch_acc = 0
                for j in range(batch_num):
                    train_data, train_label = sess.run([batch_train_data, batch_train_label])
                    train_data = np.reshape(train_data, newshape=[-1, 227, 227, 3])
                    train_label = train_label.astype(int)

                    feed = {inputs: train_data, targets: train_label}
                    batch_loss, batch_acc, opt = sess.run([loss, acc, optimizer], feed_dict=feed)
                    print(datetime.datetime.now().strftime('%c'), ' i:', i, 'j:', j, ' batch_loss:', batch_loss)
                    epoch_loss += batch_loss
                    epoch_acc += batch_acc
                epoch_loss /= batch_num
                epoch_acc /= batch_num
                print(datetime.datetime.now().strftime('%c'), ' i:', i, 'step:', step, ' epoch_loss:', epoch_loss, ' epoch_acc:', epoch_acc)
                step += 1
            coord.request_stop()
            coord.join(threads)
            sess.close()


config = Config()
network = Alexnet(config)
network.train(20)









