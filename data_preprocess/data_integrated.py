import skimage.io
from skimage import img_as_float
import os
import numpy as np
from data_preprocess.constant import *
import tensorflow as tf


class Dog:
    def __init__(self, base_dir):
        self.tfrecord_for_train_name = "dog_for_train"
        self.tfrecord_for_valid_name = "dog_for_valid"
        self.base_dir = base_dir
        self.n_class = 100
        self.dog_for_train_file = self._tfrecord_write_train()
        self.dog_for_valid_file = self._tfrecord_write_valid()

        self.train_data, self.train_label = self._tfrecord_read_train()
        self.valid_data, self.valid_label = self._tfrecord_read_valid()

    def _tfrecord_write_train(self):
        # if os.path.exists(tfrecords_dir_raw_scale + self.tfrecord_for_train_name):
        #     return tfrecords_dir_raw_scale + self.tfrecord_for_train_name
        # train_dir = self.base_dir + "train\\resized_train_3\\"
        # writer = tf.python_io.TFRecordWriter(tfrecords_dir_raw_scale + self.tfrecord_for_train_name)
        # for label in range(100):
        #     cate_dir = train_dir + str(label)
        #     for f in os.listdir(cate_dir):
        #         filename = cate_dir + "\\" + f
        #         image = img_as_float(skimage.io.imread(filename))
        #
        #         image = np.reshape(image, newshape=[-1])
        #
        #         example = tf.train.Example(features=tf.train.Features(feature={
        #             "data": tf.train.Feature(float_list=tf.train.FloatList(value=image)),
        #             "label": tf.train.Feature(float_list=tf.train.FloatList(value=[label]))
        #         }))
        #         writer.write(example.SerializeToString())
        #
        # writer.close()

        if os.path.exists(tfrecords_dir_cut_mid + self.tfrecord_for_train_name):
            return tfrecords_dir_cut_mid + self.tfrecord_for_train_name
        train_dir = self.base_dir + "train\\resized_train\\"
        writer = tf.python_io.TFRecordWriter(tfrecords_dir_cut_mid + self.tfrecord_for_train_name)
        for label in range(100):
            cate_dir = train_dir + str(label)
            for f in os.listdir(cate_dir):
                filename = cate_dir + "\\" + f
                image = img_as_float(skimage.io.imread(filename))

                image = np.reshape(image, newshape=[-1])

                example = tf.train.Example(features=tf.train.Features(feature={
                    "data": tf.train.Feature(float_list=tf.train.FloatList(value=image)),
                    "label": tf.train.Feature(float_list=tf.train.FloatList(value=[label]))
                }))
                writer.write(example.SerializeToString())
        writer.close()

        return tfrecords_dir_raw_scale + self.tfrecord_for_train_name

    def _tfrecord_write_valid(self):
        # if os.path.exists(tfrecords_dir_raw_scale + self.tfrecord_for_valid_name):
        #     return tfrecords_dir_raw_scale + self.tfrecord_for_valid_name
        # valid_dir = self.base_dir + "val\\resized_valid_3\\"
        # writer = tf.python_io.TFRecordWriter(tfrecords_dir_raw_scale + self.tfrecord_for_valid_name)
        # for label in range(100):
        #     cate_dir = valid_dir + str(label)
        #     for f in os.listdir(cate_dir):
        #         filename = cate_dir + "\\" + f
        #         image = img_as_float(skimage.io.imread(filename))
        #
        #         image = np.reshape(image, newshape=[-1])
        #         example = tf.train.Example(features=tf.train.Features(feature={
        #             "data": tf.train.Feature(float_list=tf.train.FloatList(value=image)),
        #             "label": tf.train.Feature(float_list=tf.train.FloatList(value=[label]))
        #         }))
        #         writer.write(example.SerializeToString())
        #
        # writer.close()

        if os.path.exists(tfrecords_dir_cut_mid + self.tfrecord_for_valid_name):
            return tfrecords_dir_cut_mid + self.tfrecord_for_valid_name
        valid_dir = self.base_dir + "val\\resized_valid\\"
        writer = tf.python_io.TFRecordWriter(tfrecords_dir_cut_mid + self.tfrecord_for_valid_name)
        for label in range(100):
            cate_dir = valid_dir + str(label)
            for f in os.listdir(cate_dir):
                print(label, f)
                filename = cate_dir + "\\" + f
                image = img_as_float(skimage.io.imread(filename))

                image = np.reshape(image, newshape=[-1])
                example = tf.train.Example(features=tf.train.Features(feature={
                    "data": tf.train.Feature(float_list=tf.train.FloatList(value=image)),
                    "label": tf.train.Feature(float_list=tf.train.FloatList(value=[label]))
                }))
                writer.write(example.SerializeToString())

        writer.close()

        return tfrecords_dir_raw_scale + self.tfrecord_for_valid_name

    def _tfrecord_read_train(self):
        reader = tf.TFRecordReader()
        filename_queue = tf.train.string_input_producer([self.dog_for_train_file])
        _, serialized_example = reader.read(filename_queue)
        features = tf.parse_single_example(
            serialized_example,
            features={
                "data": tf.FixedLenFeature([227 * 227 * 3], tf.float32),
                "label": tf.FixedLenFeature([1], tf.float32)
            }
        )

        data = features["data"]
        label = features["label"]
        return data, label

    def _tfrecord_read_valid(self):
        reader = tf.TFRecordReader()
        filename_queue = tf.train.string_input_producer([self.dog_for_valid_file])
        _, serialized_example = reader.read(filename_queue)
        features = tf.parse_single_example(
            serialized_example,
            features={
                "data": tf.FixedLenFeature([227 * 227 * 3], tf.float32),
                "label": tf.FixedLenFeature([1], tf.float32)
            }
        )

        data = features["data"]
        label = features["label"]
        return data, label


# d = Dog(base_dir)
# train_data_raw, train_label_raw = d.train_data, d.train_label
# batch_train_data, batch_train_label = tf.train.batch([train_data_raw, train_label_raw],
#                                                      batch_size=8,
#                                                      capacity=8153,
#                                                      num_threads=1)
# sess_config = tf.ConfigProto()
# sess_config.gpu_options.allow_growth = True
# with tf.Session(config=sess_config) as sess:
#     coord = tf.train.Coordinator()
#     threads = tf.train.start_queue_runners(sess=sess, coord=coord)
#     train_data, train_label = sess.run([batch_train_data, batch_train_label])
#     print("haha")
#     coord.request_stop()
#     coord.join(threads)
#     sess.close()



