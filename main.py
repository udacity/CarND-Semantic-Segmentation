import time
import datetime
import os.path
import tensorflow as tf
import warnings
from distutils.version import LooseVersion

import helper
import project_tests as tests
import config


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
    vgg_tag = 'vgg16'
    vgg_input_tensor_name = 'image_input:0'
    vgg_keep_prob_tensor_name = 'keep_prob:0'
    vgg_layer3_out_tensor_name = 'layer3_out:0'
    vgg_layer4_out_tensor_name = 'layer4_out:0'
    vgg_layer7_out_tensor_name = 'layer7_out:0'

    # Use tf.saved_model.loader.load to load the model and weights
    tf.saved_model.loader.load(sess, [vgg_tag], vgg_path)
    graph = tf.get_default_graph()
    input_tensor = graph.get_tensor_by_name(vgg_input_tensor_name)
    keep_prob = graph.get_tensor_by_name(vgg_keep_prob_tensor_name)
    layer3 = graph.get_tensor_by_name(vgg_layer3_out_tensor_name)
    layer4 = graph.get_tensor_by_name(vgg_layer4_out_tensor_name)
    layer7 = graph.get_tensor_by_name(vgg_layer7_out_tensor_name)

    return input_tensor, keep_prob, layer3, layer4, layer7
tests.test_load_vgg(load_vgg, tf)


def conv2d(input, filters, kernel_size, strides):
    return tf.layers.conv2d(input, filters, kernel_size=kernel_size, strides=strides,
                            padding='SAME', kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3))

def conv2d_transpose(input, filters, kernel_size, strides):
    return tf.layers.conv2d_transpose(input, filters, kernel_size=kernel_size, strides=strides,
                                      padding='SAME', kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3))


def layers(vgg_layer3_out, vgg_layer4_out, vgg_layer7_out, num_classes):
    """
    Create the layers for a fully convolutional network. Build skip-layers using the vgg layers.
    :param vgg_layer7_out: TF Tensor for VGG Layer 3 output
    :param vgg_layer4_out: TF Tensor for VGG Layer 4 output
    :param vgg_layer3_out: TF Tensor for VGG Layer 7 output
    :param num_classes: Number of classes to classify
    :return: The Tensor for the last layer of output
    """

    l7_conv1x1 = conv2d(vgg_layer7_out, num_classes, 1, 1)

    # deconvolution layers
    transposed_l7_output = conv2d_transpose(l7_conv1x1, num_classes, 8, 4)
    transposed_l4_output = conv2d_transpose(vgg_layer4_out, num_classes, 4, 2)
    transposed_l3_output = conv2d_transpose(vgg_layer3_out, num_classes, 1, 1)

    output = tf.add(transposed_l7_output, transposed_l4_output)
    output = tf.add(output, transposed_l3_output)
    output = conv2d_transpose(output, num_classes, 16, 8)

    return output
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
    # Implement function
    logits = tf.reshape(nn_last_layer, (-1, num_classes))
    labels = tf.reshape(correct_label, (-1, num_classes))

    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels))
    train_op = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

    return logits, train_op, loss
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

    for epoch in range(0, epochs):

        total_training_loss = 0

        for image, label in get_batches_fn(batch_size):
            start = time.time()

            _, loss = sess.run([train_op, cross_entropy_loss],
                                             feed_dict={input_image: image, correct_label: label,
                                                        keep_prob: 0.5, learning_rate: 0.001})

            total_training_loss += loss
            end = time.time()
            print("elapsed time:{0}".format(end - start))

        message = "epoch:{}, total_training_loss:{}".format(epoch+1, total_training_loss)
        print(message)

tests.test_train_nn(train_nn)


def run():
    # num_classes = 2
    # image_shape = (160, 576)
    # data_dir = './data'
    # runs_dir = './runs'
    # model_dir = './model'

    num_classes = config.training_options["num_class"]
    image_shape = config.training_options["image_shape"]
    data_dir = config.data_paths["data_dir"]
    runs_dir = config.data_paths["runs_dir"]
    model_dir = config.data_paths["model_dir"]

    epochs = 1
    batch_size = 10

    # check for dataset
    tests.test_for_kitti_dataset(data_dir)

    # Download pretrained vgg model
    helper.maybe_download_pretrained_vgg(data_dir)

    # Save model
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)

    model_save_path = os.path.join(model_dir, str(datetime.datetime.now()).replace(' ', '-'))

    # OPTIONAL: Train and Inference on the cityscapes dataset instead of the Kitti dataset.
    # You'll need a GPU with at least 10 teraFLOPS to train on.
    #  https://www.cityscapes-dataset.com/

    with tf.Session() as sess:
        # Path to vgg model
        vgg_path = os.path.join(data_dir, 'vgg')
        # Create function to get batches
        get_batches_fn = helper.gen_batch_function(os.path.join(data_dir, 'data_road/training'), image_shape)
        correct_label = tf.placeholder(tf.int32, (None, image_shape[0], image_shape[1], num_classes))
        learning_rate = tf.placeholder(tf.float32, None)

        # OPTIONAL: Augment Images for better results
        #  https://datascience.stackexchange.com/questions/5224/how-to-prepare-augment-images-for-neural-network

        # Build NN using load_vgg, layers, and optimize function
        input_image, keep_prob, layer3_out, layer4_out, layer7_out = load_vgg(sess, vgg_path)

        layer_output = layers(layer3_out, layer4_out, layer7_out, num_classes)
        logits, train_op, loss = optimize(layer_output, correct_label, learning_rate, num_classes)

        saver = tf.train.Saver()

        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())

        # TODO: Train NN using the train_nn function
        train_nn(sess, epochs, batch_size, get_batches_fn, train_op, loss,
                 input_image, correct_label, keep_prob, learning_rate)

        # TODO: Save inference data using helper.save_inference_samples
        helper.save_inference_samples(runs_dir, data_dir, sess, image_shape, logits, keep_prob, input_image)

        saver.save(sess, model_save_path)

        # OPTIONAL: Apply the trained model to a video


if __name__ == '__main__':
    run()
