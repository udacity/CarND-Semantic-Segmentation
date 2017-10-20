import tensorflow as tf
import scipy.misc
import numpy as np

import project_tests as tests


def load_vgg(sess, vgg_path):
    """
    Load Pretrained VGG Model into TensorFlow.
    :param sess: TensorFlow Session
    :param vgg_path: Path to vgg folder, containing "variables/" and "saved_model.pb"
    :return: Tuple of Tensors from VGG model (image_input, keep_prob, layer3_out, layer4_out, layer7_out)
    """
    # TODO: Implement function
    # Use tf.saved_model.loader.load to load the model and weights
    vgg_tag = 'vgg16'
    vgg_input_tensor_name = 'image_input:0'
    vgg_keep_prob_tensor_name = 'keep_prob:0'
    vgg_layer3_out_tensor_name = 'layer3_out:0'
    vgg_layer4_out_tensor_name = 'layer4_out:0'
    vgg_layer7_out_tensor_name = 'layer7_out:0'

    # タグ指定でモデル読み込み
    tf.saved_model.loader.load(sess, [vgg_tag], vgg_path)

    graph = tf.get_default_graph()
    image_input = graph.get_tensor_by_name(vgg_input_tensor_name)
    keep_prob_tensor = graph.get_tensor_by_name(vgg_keep_prob_tensor_name)
    layer3_out = graph.get_tensor_by_name(vgg_layer3_out_tensor_name)
    layer4_out = graph.get_tensor_by_name(vgg_layer4_out_tensor_name)
    layer7_out = graph.get_tensor_by_name(vgg_layer7_out_tensor_name)

    return image_input, keep_prob_tensor, layer3_out, layer4_out, layer7_out


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
    # TODO: Implement function
    # l7_conv1x1 = tf.layers.conv2d(vgg_layer7_out, num_classes, 1, 1, padding='SAME',
    #                               kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3))
    l7_conv1x1 = conv2d(vgg_layer7_out, num_classes, 1, 1)

    # deconvolution layers
    # transposed_l7_output = tf.layers.conv2d_transpose(l7_conv1x1, num_classes, 8, 4, padding='SAME',
    #                                                   kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3))
    transposed_l7_output = conv2d_transpose(l7_conv1x1, num_classes, 8, 4)
    # tf.Print(transposed_l7_output, [tf.shape(transposed_l7_output)])

    # transposed_l4_output = tf.layers.conv2d_transpose(vgg_layer4_out, num_classes, 4, 2, padding='SAME',
    #                                                   kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3))
    transposed_l4_output = conv2d_transpose(vgg_layer4_out, num_classes, 4, 2)
    # tf.Print(transposed_l4_output, [tf.shape(transposed_l4_output)])

    # transposed_l3_output = tf.layers.conv2d(vgg_layer3_out, num_classes, 1, 1, padding='SAME',
    #                                         kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3))
    transposed_l3_output = conv2d_transpose(vgg_layer3_out, num_classes, 1, 1)

    # tf.Print(transposed_l3_output, [tf.shape(transposed_l3_output)])

    output = tf.add(transposed_l7_output, transposed_l4_output)
    output = tf.add(output, transposed_l3_output)
    # output = tf.layers.conv2d_transpose(output, num_classes, 16, 8, padding='SAME',
    #                                     kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3))

    output = conv2d_transpose(output, num_classes, 16, 8)

    return output
tests.test_layers(layers)


def load_image(img_path, img_shape):
    return scipy.misc.imresize(scipy.misc.imread(img_path), img_shape)


def main():

    imgs = []
    img = load_image("./data/data_road/testing/image_2/um_000000.png", (160, 576))
    imgs.append(img)
    input_img = np.array(imgs)
    print(input_img.shape)

    with tf.Session() as sess:

        vgg_path = "./data/vgg/"
        input_tensor, keep_prob, l3, l4, l7 = load_vgg(sess, vgg_path)

        # l3out, l4out, l7out = sess.run([l3, l4, l7], feed_dict={input_tensor: input_img, keep_prob: 0.5})
        # print(l3out.shape)
        # print(l4out.shape)
        # print(l7out.shape)

        output = layers(l3, l4, l7, 2)

        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())

        result = sess.run([output], feed_dict={input_tensor: input_img, keep_prob: 0.5})


        # print(type(result))
        print(result[0][0].shape)


if __name__ == "__main__":
    main()