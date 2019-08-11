#!/usr/bin/env python3
import os.path
import tensorflow as tf
import helper
import warnings
from distutils.version import LooseVersion
import project_tests as tests
import numpy as np
import matplotlib.pyplot as plt
import math


# Check TensorFlow Version
assert LooseVersion(tf.__version__) >= LooseVersion('1.0'), 'Please use TensorFlow version 1.0 or newer.  You are using {}'.format(tf.__version__)
print('TensorFlow Version: {}'.format(tf.__version__))

# Check for a GPU
if not tf.test.gpu_device_name():
  warnings.warn('No GPU found. Please use a GPU to train your neural network.')
else:
  print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))


def load_vgg(sess, vgg_path):
  """
  Load Pretrained VGG Model into TensorFlow.
  :param sess: TensorFlow Session
  :param vgg_path: Path to vgg folder, containing "variables/" and "saved_model.pb"
  :return: Tuple of Tensors from VGG model (image_input, keep_prob, layer3_out, layer4_out, layer7_out)
  """
  # TODO: Implement function
  #   Use tf.saved_model.loader.load to load the model and weights
  vgg_tag = 'vgg16'
  vgg_input_tensor_name = 'image_input:0'
  vgg_keep_prob_tensor_name = 'keep_prob:0'
  vgg_layer3_out_tensor_name = 'layer3_out:0'
  vgg_layer4_out_tensor_name = 'layer4_out:0'
  vgg_layer7_out_tensor_name = 'layer7_out:0'

  tf.saved_model.loader.load(sess, [vgg_tag], vgg_path)
  graph = tf.get_default_graph()
  input_layer = graph.get_tensor_by_name(vgg_input_tensor_name)
  keep_prob = graph.get_tensor_by_name(vgg_keep_prob_tensor_name)
  layer_3 = graph.get_tensor_by_name(vgg_layer3_out_tensor_name)
  layer_4 = graph.get_tensor_by_name(vgg_layer4_out_tensor_name)
  layer_7 = graph.get_tensor_by_name(vgg_layer7_out_tensor_name)
  
  return input_layer, keep_prob, layer_3, layer_4, layer_7
tests.test_load_vgg(load_vgg, tf)


def layers(vgg_layer3_out, vgg_layer4_out, vgg_layer7_out, num_classes):
  """
  Create the layers for a fully convolutional network.  Build skip-layers using the vgg layers.
  :param vgg_layer3_out: TF Tensor for VGG Layer 3 output
  :param vgg_layer4_out: TF Tensor for VGG Layer 4 output
  :param vgg_layer7_out: TF Tensor for VGG Layer 7 output
  :param num_classes: Number of classes to classify
  :return: The Tensor for the last layer of output
  """
  # 1x1 Convolution applied to VGG layer 7
  vgg_7_1x1 = tf.layers.conv2d(vgg_layer7_out, num_classes, 1, padding = 'same',
        kernel_initializer = tf.random_normal_initializer(stddev = 0.01),
        #kernel_initializer = tf.random_uniform_initializer(minval = 0.001, maxval = 0.001),
        kernel_regularizer = tf.contrib.layers.l2_regularizer(1e-3))
  # Upsampling the output of the 1x1 Convolution
  vgg_7_upsample = tf.layers.conv2d_transpose(vgg_7_1x1, num_classes, 4,
        strides = (2,2), padding = 'same',
        kernel_initializer = tf.random_normal_initializer(stddev=0.01),
        #kernel_initializer = tf.random_uniform_initializer(minval = 0.001, maxval = 0.001),
        kernel_regularizer = tf.contrib.layers.l2_regularizer(1e-3))
  # 1x1 Convolution applied to VGG layer 4
  vgg_4_1x1 = tf.layers.conv2d(vgg_layer4_out, num_classes, 1, padding = 'same',
        kernel_initializer= tf.random_normal_initializer(stddev=0.01),
        #kernel_initializer = tf.random_uniform_initializer(minval = 0.001, maxval = 0.001),
        kernel_regularizer = tf.contrib.layers.l2_regularizer(1e-3))
  # Connecting the output from the upsampled VGG layer 7 and VGG layer 4
  connect_7_4 = tf.add(vgg_7_upsample, vgg_4_1x1)
  # Usampling the output from merging VGG Layers 7 and 4
  vgg_7_4_upsample = tf.layers.conv2d_transpose(connect_7_4, num_classes, 4,
        strides = (2,2), padding = 'same',
        kernel_initializer = tf.random_normal_initializer(stddev=0.01),
        #kernel_initializer = tf.random_uniform_initializer(minval = 0.001, maxval = 0.001),
        kernel_regularizer = tf.contrib.layers.l2_regularizer(1e-3))
  # 1x1 Convolution applied to VGG layer 3
  vgg_3_1x1 = tf.layers.conv2d(vgg_layer3_out, num_classes, 1, padding = 'same',
        kernel_initializer = tf.random_normal_initializer(stddev=0.01),
        #kernel_initializer = tf.random_uniform_initializer(minval = 0.001, maxval = 0.001),
        kernel_regularizer = tf.contrib.layers.l2_regularizer(1e-3))
  # Connecting the VGG layers 3, 4, and 7
  connect_7_4_3 = tf.add(vgg_7_4_upsample, vgg_3_1x1)
  # Upsampling the output of the last skip connection
  nn_last_layer = tf.layers.conv2d_transpose(connect_7_4_3, num_classes, 16, 8,
        padding = 'same',
        kernel_initializer = tf.random_normal_initializer(stddev=0.01),
        #kernel_initializer = tf.random_uniform_initializer(minval = 0.001, maxval = 0.001),
        kernel_regularizer = tf.contrib.layers.l2_regularizer(1e-3))

  return nn_last_layer
tests.test_layers(layers)


def optimize(nn_last_layer, correct_label, learning_rate, num_classes):
  """
  Build the TensorFLow loss and optimizer operations.
  :param nn_last_layer: TF Tensor of the last layer in the neural network
  :param correct_label: TF Placeholder for the correct label image
  :param learning_rate: TF Placeholder for the learning rate
  :param num_classes: Number of classes to classify
  :return: Tuple of (logits, train_op, cross_entropy_loss)
  """
  # Creating logits amd ground truth which are each 2D tensors with the same
  # dimensions as the last layer
  logits = tf.reshape(nn_last_layer, (-1, num_classes))
  correct_label = tf.reshape(correct_label, (-1, num_classes))
  # Using a Softmax Cross Entropy loss function
  loss_function = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
                                 logits = logits, labels = correct_label))
  # Using Adam Optimizer
  opt = tf.train.AdamOptimizer(learning_rate = learning_rate)
  train = opt.minimize(loss_function)

  return logits, train, loss_function
tests.test_optimize(optimize)


def train_nn(sess, epochs, batch_size, get_batches_fn, train_op, cross_entropy_loss, input_image,
             correct_label, keep_prob, learning_rate):
  """
  Train neural network and print out the loss during training.
  :param sess: TF Session
  :param epochs: Number of epochs
  :param batch_size: Batch size
  :param get_batches_fn: Function to get batches of training data.  Call using get_batches_fn(batch_size)
  :param train_op: TF Operation to train the neural network
  :param cross_entropy_loss: TF Tensor for the amount of loss
  :param input_image: TF Placeholder for input images
  :param correct_label: TF Placeholder for label images
  :param keep_prob: TF Placeholder for dropout keep probability
  :param learning_rate: TF Placeholder for learning rate
  """
  # TODO: Implement function
  # Initialize
  sess.run(tf.global_variables_initializer())
  detailed_loss = []
  epoch_loss = []
  print("------------Begin Training----------------")
  for epoch in range(epochs):
    print("Beginning Epoch ", epoch + 1)
    for image, label in get_batches_fn(batch_size):
      _, loss = sess.run([train_op, cross_entropy_loss],
                          feed_dict={input_image: image, correct_label: label,
                                     keep_prob: 0.5, learning_rate: 0.0001})
      detailed_loss.append(loss)
      print("Training Loss: ", loss)
    epoch_loss.append(loss)
  
  detailed_axis = range(1, (len(detailed_loss) + 1))
  epoch_axis = range(1, (len(epoch_loss) + 1))

  plt.subplot(3, 1, 1)
  plt.plot(epoch_axis, epoch_loss)
  plt.ylim(0, math.ceil(max(epoch_loss)))
  plt.title('Epoch Loss - Batch: 10, Learning Rate: 0.0001, Random Normal')
  plt.xlabel('Epoch')
  plt.ylabel('Loss')

  plt.subplot(3, 1, 3)
  plt.plot(detailed_axis, detailed_loss)
  plt.ylim(0, math.ceil(max(detailed_loss)))
  plt.title('Batch Loss - Batch: 10, Learning Rate 0.0001, Random Normal')
  plt.xlabel('Batch')
  plt.ylabel('Loss')
  plt.savefig('loss_b10_e50_RN_LR0001.png')
tests.test_train_nn(train_nn)


def run():
  num_classes = 2
  image_shape = (160, 576)  # KITTI dataset uses 160x576 images
  data_dir = './data'
  runs_dir = './runs'
  tests.test_for_kitti_dataset(data_dir)

  # Download pretrained vgg model
  helper.maybe_download_pretrained_vgg(data_dir)

  # OPTIONAL: Train and Inference on the cityscapes dataset instead of the Kitti dataset.
  # You'll need a GPU with at least 10 teraFLOPS to train on.
  #  https://www.cityscapes-dataset.com/

  with tf.Session() as sess:
    # Path to vgg model
    vgg_path = os.path.join(data_dir, 'vgg')
    # Create function to get batches
    get_batches_fn = helper.gen_batch_function(os.path.join(data_dir, 'data_road/training'), image_shape)

    # OPTIONAL: Augment Images for better results
    #  https://datascience.stackexchange.com/questions/5224/how-to-prepare-augment-images-for-neural-network

    # TODO: Build NN using load_vgg, layers, and optimize function

    num_epochs = 50
    batch_size = 10

    correct_label = tf.placeholder(tf.int32, [None, None, None, num_classes], name = 'correct_label')
    learning_rate = tf.placeholder(tf.float32, name = 'learning_rate')

    input_image, keep_prob, layer_3, layer_4, layer_7 = load_vgg(sess, vgg_path)
    layer_output = layers(layer_3, layer_4, layer_7, num_classes)
    logits, train_op, cross_entropy_loss = optimize(layer_output, correct_label, learning_rate, num_classes)

        
    # TODO: Train NN using the train_nn function
    train_nn(sess, num_epochs, batch_size, get_batches_fn, train_op, cross_entropy_loss,
             input_image, correct_label, keep_prob, learning_rate)

    # TODO: Save inference data using helper.save_inference_samples
    helper.save_inference_samples(runs_dir, data_dir, sess, image_shape, logits, keep_prob, input_image)
    # OPTIONAL: Apply the trained model to a video


if __name__ == '__main__':
    run()
