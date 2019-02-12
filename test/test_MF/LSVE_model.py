import tensorflow as tf
import numpy as np


def model_single(inputLowData_tensor):
    #    with tf.device("/gpu:0"):
    input_current = inputLowData_tensor  # lowData

    # due to don't have training_Set at right now, so let it be annotation.
    tensor = None

    # ----------------------------------------------Frame  0--------------------------------------------------------
    input_current_w = tf.get_variable("input_current_w", [5, 5, 1, 64],
                                initializer=tf.random_normal_initializer(stddev=np.sqrt(2.0 / 25)))
    input_current_b = tf.get_variable("input_current_b", [64], initializer=tf.constant_initializer(0))
    input_low_tensor = tf.nn.relu(
        tf.nn.bias_add(tf.nn.conv2d(input_current, input_current_w, strides=[1, 1, 1, 1], padding='SAME'),
                       input_current_b))

    # ----------------------------------1x1 conv, for reduce number of parameters----------------
    input_3x3_w = tf.get_variable("input_1x1_w", [3, 3, 64, 64],
                                     initializer=tf.random_normal_initializer(stddev=np.sqrt(2.0 / 64 / 9)))
    input_1x1_b = tf.get_variable("input_1x1_b", [64], initializer=tf.constant_initializer(0))
    input_3x3_tensor = tf.nn.relu(
        tf.nn.bias_add(tf.nn.conv2d(input_low_tensor, input_3x3_w, strides=[1, 1, 1, 1], padding='SAME'),
                       input_1x1_b))
    tensor = input_3x3_tensor

    # --------------------------------------start iteration for last layers----------------------------------
    convId = 0
    for i in range(18):
        conv_w = tf.get_variable("conv_%02d_w" % (convId), [3, 3, 64, 64],
                                 initializer=tf.random_normal_initializer(stddev=np.sqrt(2.0 / 9 / 64)))
        tf.add_to_collection(tf.GraphKeys.WEIGHTS, conv_w)
        conv_b = tf.get_variable("conv_%02d_b" % (convId), [64], initializer=tf.constant_initializer(0))
        convId += 1
        tensor = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(tensor, conv_w, strides=[1, 1, 1, 1], padding='SAME'), conv_b))

    conv_w = tf.get_variable("conv_%02d_w" % (convId), [3, 3, 64, 1],
                             initializer=tf.random_normal_initializer(stddev=np.sqrt(2.0 / 9 / 64)))
    conv_b = tf.get_variable("conv_%02d_b" % (convId), [1], initializer=tf.constant_initializer(0))
    convId += 1
    tf.add_to_collection(tf.GraphKeys.WEIGHTS, conv_w)
    tensor = tf.nn.bias_add(tf.nn.conv2d(tensor, conv_w, strides=[1, 1, 1, 1], padding='SAME'), conv_b)
    tensor = tf.add(tensor, input_current)
    return tensor
