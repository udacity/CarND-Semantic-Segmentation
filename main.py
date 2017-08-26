import os.path
import tensorflow as tf
import numpy as np
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
    # TODO: Implement function
    #   Use tf.saved_model.loader.load to load the model and weights
    vgg_tag = 'vgg16'
    vgg_input_tensor_name = 'image_input:0'
    vgg_keep_prob_tensor_name = 'keep_prob:0'
    vgg_layer3_out_tensor_name = 'layer3_out:0'
    vgg_layer4_out_tensor_name = 'layer4_out:0'
    vgg_layer7_out_tensor_name = 'layer7_out:0'

    tf.saved_model.loader.load(sess, [vgg_tag], vgg_path)

    input_image = sess.graph.get_tensor_by_name(vgg_input_tensor_name)
    keep_prob   = sess.graph.get_tensor_by_name(vgg_keep_prob_tensor_name)
    layer3_out  = sess.graph.get_tensor_by_name(vgg_layer3_out_tensor_name)
    layer4_out  = sess.graph.get_tensor_by_name(vgg_layer4_out_tensor_name)
    layer7_out  = sess.graph.get_tensor_by_name(vgg_layer7_out_tensor_name)

    return input_image, keep_prob, layer3_out, layer4_out, layer7_out
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
    # TODO: Implement function
    # VGG Encoder: Input image: 160x576x3
    # Output of 1st conv: 160x576x64
    # Output of 2nd conv: 80x288x128
    # Output of 3rd conv: 40x144x256     (= vgg_layer3_out)
    # Output of 4th conv: 20x72x512      (= vgg_layer4_out)
    # Output of 5th conv: 10x36x512
    # Output of last max pooling: 5x18x512

    kernel_initializer = tf.truncated_normal_initializer(stddev=0.01)

    # Encoder
    encoder_output = tf.layers.conv2d(vgg_layer7_out, 4096, kernel_size=(1,1), strides=(1, 1))
#    encoder_output = tf.layers.conv2d(vgg_layer7_out, 4096, kernel_size=(1,1), strides=(1, 1),
#                                      kernel_initializer=kernel_initializer)        # (5x18x4096)

    # Decoder
#    decoder1 = tf.layers.conv2d_transpose(encoder_output, 512, kernel_size=(2,2), strides=(2,2),
#                                          kernel_initializer=kernel_initializer)    # (10x36x512)
#    decoder2 = tf.layers.conv2d_transpose(decoder1, 512, kernel_size=(2,2), strides=(2,2),
#                                          kernel_initializer=kernel_initializer)    # (20x72x512)
#    decoder2 = tf.add(decoder2, vgg_layer4_out)
#    decoder3 = tf.layers.conv2d_transpose(decoder2, 256, kernel_size=(2,2), strides=(2,2),
#                                          kernel_initializer=kernel_initializer)    # (40x144x256)
#    decoder3 = tf.add(decoder3, vgg_layer3_out)
#    decoder4 = tf.layers.conv2d_transpose(decoder3, 128, kernel_size=(2,2), strides=(2,2),
#                                          kernel_initializer=kernel_initializer)    # (80x288x128)
#    decoder5 = tf.layers.conv2d_transpose(decoder4, 64, kernel_size=(2,2), strides=(2,2),
#                                          kernel_initializer=kernel_initializer)    # (160x576x64)
#    decoder6 = tf.layers.conv2d_transpose(decoder5, num_classes, kernel_size=(1,1), strides=(1,1),
#                                          kernel_initializer=kernel_initializer)    # (160x576xnum_classes)
#    decoder7 = tf.layers.conv2d_transpose(decoder6, num_classes, kernel_size=(1,1), strides=(1,1),
#                                          kernel_initializer=kernel_initializer)
    decoder7 = tf.layers.conv2d_transpose(encoder_output, num_classes, kernel_size=(1,1), strides=(1,1))

    return decoder7
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
    # TODO: Implement function
    logits = tf.reshape(nn_last_layer, (-1, num_classes))
    cross_entropy_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=correct_label, logits=logits))
    training_operation = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(cross_entropy_loss)

    return logits, training_operation, cross_entropy_loss
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

    # Hyperparameters
    lr = 0.001     # learning_rate
    kp = 0.5       # keep_prob

    print()
    print('Training...')
    print('vars not initialized', sess.run(tf.report_uninitialized_variables()))
    for epoch in range(epochs):
        gen = get_batches_fn(batch_size)
        for images, gt_images in gen:
#            sess.run(train_op, feed_dict={input_image: images, correct_label: gt_images, keep_prob: kp, learning_rate: lr})
            _, loss = sess.run([train_op, cross_entropy_loss],
                               feed_dict={input_image: images,
                                          correct_label: gt_images,
                                          keep_prob: kp,
                                          learning_rate: lr}
                              )
            print('Epoch {}: loss = {}'.format(epoch, loss))

tests.test_train_nn(train_nn)

###TEST ZONE



def run():
    # Hyperparameters
    epochs = 1
    batch_size = 1

    # Placeholders
    correct_label = tf.placeholder(tf.float32)    # not bool?
    keep_prob = tf.placeholder(tf.float32)
    learning_rate = tf.placeholder(tf.float32)

    # Variables
    num_classes = 2          # classes: road, not road
#    image_shape = (160, 576)
    image_shape = (40, 144)
    data_dir = './data'
    runs_dir = './runs'
    tests.test_for_kitti_dataset(data_dir)

    # Download pretrained vgg model
    helper.maybe_download_pretrained_vgg(data_dir)

    # OPTIONAL: Train and Inference on the cityscapes dataset instead of the Kitti dataset.
    # You'll need a GPU with at least 10 teraFLOPS to train on.
    #  https://www.cityscapes-dataset.com/

### TEST ZONE
    

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        #sess.run(tf.local_variables_initializer())

###     TEST ZONE
        


        # Path to vgg model
        vgg_path = os.path.join(data_dir, 'vgg')
        # Create function to get batches
        get_batches_fn = helper.gen_batch_function(os.path.join(data_dir, 'data_road/training'), image_shape)

        # OPTIONAL: Augment Images for better results
        #  https://datascience.stackexchange.com/questions/5224/how-to-prepare-augment-images-for-neural-network

        # TODO: Build NN using load_vgg, layers, and optimize function
        input_image, keep_prob, vgg_layer3_out, vgg_layer4_out, vgg_layer7_out = load_vgg(sess, vgg_path)
        nn_last_layer = layers(vgg_layer3_out, vgg_layer4_out, vgg_layer7_out, num_classes)
        logits, train_operation, cross_entropy_loss = optimize(nn_last_layer, correct_label, learning_rate, num_classes)
        
        # TODO: Train NN using the train_nn function
        train_nn(sess, epochs, batch_size, get_batches_fn, train_operation, cross_entropy_loss, input_image, correct_label,
                 keep_prob, learning_rate)

        # TODO: Save inference data using helper.save_inference_samples
        #  helper.save_inference_samples(runs_dir, data_dir, sess, image_shape, logits, keep_prob, input_image)

        # OPTIONAL: Apply the trained model to a video


if __name__ == '__main__':
    run()
