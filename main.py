#!/usr/bin/env python3
import os.path
import tensorflow as tf
import helper
import warnings
from distutils.version import LooseVersion
import project_tests as tests


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
    tf.saved_model.loader.load(sess, ['vgg16'], vgg_path)

    graph = tf.get_default_graph()

    vgg_input = graph.get_tensor_by_name('image_input:0')
    vgg_keep_prob = graph.get_tensor_by_name('keep_prob:0')
    vgg_layer3_out = graph.get_tensor_by_name('layer3_out:0')
    vgg_layer4_out = graph.get_tensor_by_name('layer4_out:0')
    vgg_layer7_out = graph.get_tensor_by_name('layer7_out:0')

    return (
        vgg_input,
        vgg_keep_prob,
        vgg_layer3_out,
        vgg_layer4_out,
        vgg_layer7_out,
    )

# tests.test_load_vgg(load_vgg, tf)


def layers(vgg_layer3_out, vgg_layer4_out, vgg_layer7_out, num_classes):
    """
    Create the layers for a fully convolutional network.  Build skip-layers using the vgg layers.
    :param vgg_layer3_out: TF Tensor for VGG Layer 3 output
    :param vgg_layer4_out: TF Tensor for VGG Layer 4 output
    :param vgg_layer7_out: TF Tensor for VGG Layer 7 output
    :param num_classes: Number of classes to classify
    :return: The Tensor for the last layer of output
    """
    # "This is where all the fun happens"

    l2_reg = tf.contrib.layers.l2_regularizer
    conv_1x1 = tf.layers.conv2d(vgg_layer7_out, num_classes,
                                kernel_size=1,
                                padding='SAME',
                                kernel_regularizer=l2_reg(1e-3))

    output = tf.layers.conv2d_transpose(conv_1x1, num_classes,
                                        kernel_size=4, strides=(2, 2),
                                        padding='SAME',
                                        kernel_regularizer=l2_reg(1e-3))

    output = tf.layers.conv2d_transpose(output, filters=16,
                                        kernel_size=4, strides=(2, 2),
                                        padding='SAME',
                                        kernel_regularizer=l2_reg(1e-3))

    output = tf.layers.conv2d_transpose(output, filters=num_classes,
                                       kernel_size=16, strides=(8, 8),
                                       padding='SAME')
    return output

# tests.test_layers(layers)


def optimize(nn_last_layer, correct_label, learning_rate, num_classes):
    """
    Build the TensorFLow loss and optimizer operations.
    :param nn_last_layer: TF Tensor of the last layer in the neural network
    :param correct_label: TF Placeholder for the correct label image
    :param learning_rate: TF Placeholder for the learning rate
    :param num_classes: Number of classes to classify
    :return: Tuple of (logits, train_op, cross_entropy_loss)
    """
    logits = tf.reshape(nn_last_layer, (-1, num_classes))
    cross_entropy_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
        labels=correct_label,
        logits=logits,
    ))
    optimizer = tf.train.AdamOptimizer(learning_rate)
    train_op = optimizer.minimize(
        loss=cross_entropy_loss,
        global_step=tf.train.get_global_step())

    return logits, train_op, cross_entropy_loss

# tests.test_optimize(optimize)


def train_nn(sess, epochs, batch_size, get_batches_fn, train_op, cross_entropy_loss,
             input_image, correct_label, keep_prob, learning_rate):
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
    for epoch in range(epochs):
        batch_num = 0
        for image, label in get_batches_fn(batch_size):
            _ = sess.run(train_op, {
                input_image : image,
                keep_prob : 0.5,
                correct_label : label,
            })
            if batch_num % 10:
                print('Done with batch number {}'.format(batch_num))
            batch_num += 1

# tests.test_train_nn(train_nn)


def run():
    num_classes = 2
    image_shape = (160, 576)
    data_dir = './data'
    runs_dir = './runs'
    epochs = 1
    batch_size = 2
    learning_rate = 1e-4
    tests.test_for_kitti_dataset(data_dir)

    # Download pretrained vgg model
    helper.maybe_download_pretrained_vgg(data_dir)

    # OPTIONAL: Train and Inference on the cityscapes dataset instead of the Kitti dataset.
    # You'll need a GPU with at least 10 teraFLOPS to train on.
    #  https://www.cityscapes-dataset.com/

    with tf.Session() as sess:
        # Create function to get batches
        get_batches_fn = helper.gen_batch_function(
            os.path.join(data_dir, 'data_road/training'),
            image_shape
        )

        # OPTIONAL: Augment Images for better results
        #  https://datascience.stackexchange.com/questions/5224/how-to-prepare-augment-images-for-neural-network

        # Build NN using load_vgg, layers, and optimize function
        vgg_path = os.path.join(data_dir, 'vgg')  # Path to vgg model
        input_tensor, keep_prob, layer3_out, layer4_out, layer7_out = load_vgg(
            sess,
            vgg_path
        )

        nn_last_layer = layers(layer3_out, layer4_out, layer7_out, num_classes)

        # Get the optimizer
        correct_label = tf.placeholder(tf.int32, [None, None, None, num_classes])
        logits, train_op, cross_entropy_loss = optimize(
            nn_last_layer,
            correct_label,
            learning_rate,
            num_classes
        )

        sess.run(tf.global_variables_initializer())
        print('Variables initialized')
        # Train NN using the train_nn function
        train_nn(
            sess,
            epochs, batch_size, get_batches_fn,
            train_op, cross_entropy_loss,
            input_tensor, correct_label,
            keep_prob, learning_rate
        )

        # TODO: Save inference data using helper.save_inference_samples
        #  helper.save_inference_samples(runs_dir, data_dir, sess, image_shape, logits, keep_prob, input_image)

        # OPTIONAL: Apply the trained model to a video


if __name__ == '__main__':
    run()
