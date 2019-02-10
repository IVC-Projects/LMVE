import tensorflow as tf
import numpy as np


def model_double(inputHigh1Data_tensor, inputLowData_tensor, inputHigh2Data_tensor):
    #    with tf.device("/gpu:0"):
    input_before = inputHigh1Data_tensor  # highData
    input_current = inputLowData_tensor  # lowData

    input_after = inputHigh2Data_tensor

    # due to don't have training_Set at right now, so let it be annotation.
    tensor = None

    # ----------------------------------------------Frame -1--------------------------------------------------------
    input_before_w = tf.get_variable("input_before_w", [5, 5, 1, 64],
                                initializer=tf.random_normal_initializer(stddev=np.sqrt(2.0 / 25)))
    input_before_b = tf.get_variable("input_before_b", [64], initializer=tf.constant_initializer(0))
    input_high1_tensor = tf.nn.relu(
        tf.nn.bias_add(tf.nn.conv2d(input_before, input_before_w, strides=[1, 1, 1, 1], padding='SAME'), input_before_b))
    # ----------------------------------------------Frame  0--------------------------------------------------------
    input_current_w = tf.get_variable("input_current_w", [5, 5, 1, 64],
                                initializer=tf.random_normal_initializer(stddev=np.sqrt(2.0 / 25)))
    input_current_b = tf.get_variable("input_current_b", [64], initializer=tf.constant_initializer(0))
    input_low_tensor = tf.nn.relu(
        tf.nn.bias_add(tf.nn.conv2d(input_current, input_current_w, strides=[1, 1, 1, 1], padding='SAME'),
                       input_current_b))

    # ----------------------------------------------Frame  1--------------------------------------------------------
    input_after_w = tf.get_variable("input_after_w", [5, 5, 1, 64],
                                initializer=tf.random_normal_initializer(stddev=np.sqrt(2.0 / 25)))
    input_after_b = tf.get_variable("input_after_b", [64], initializer=tf.constant_initializer(0))
    input_high2_tensor = tf.nn.relu(
        tf.nn.bias_add(tf.nn.conv2d(input_after, input_after_w, strides=[1, 1, 1, 1], padding='SAME'),
                       input_after_b))
    # ------------------------------------------Frame -1\0\1 concat------------------------------------------
    input_tensor_Concat = tf.concat([input_high1_tensor, input_low_tensor, input_high2_tensor], axis=3)
    # ----------------------------------1x1 conv, for reduce number of parameters----------------
    input_1x1_w = tf.get_variable("input_1x1_w", [1, 1, 192, 64],
                                     initializer=tf.random_normal_initializer(stddev=np.sqrt(2.0 / 192)))
    input_1x1_b = tf.get_variable("input_1x1_b", [64], initializer=tf.constant_initializer(0))
    input_1x1_tensor = tf.nn.relu(
        tf.nn.bias_add(tf.nn.conv2d(input_tensor_Concat, input_1x1_w, strides=[1, 1, 1, 1], padding='SAME'),
                       input_1x1_b))
    tensor = input_1x1_tensor

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
