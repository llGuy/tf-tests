import tensorflow as tf
import numpy as np
import matplotlib.pylab as plt

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot = True)

learning_rate = 0.5
epochs = 10
batch_size = 100

# The images are of dimensions 28 * 28
x = tf.placeholder(tf.float32, [None, 28 * 28])

# The output (which digit is it, but in an array - 0 for all the ones that are not, and 1 for the one that is correct)
y = tf.placeholder(tf.float32, [None, 10])

# 3.0) A Neural Network Example (https://adventuresinmachinelearning.com/python-tensorflow-tutorial/)
# 2.5) The Notation (https://adventuresinmachinelearning.com/neural-networks-tutorial/)
