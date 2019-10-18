# Semantic Segmentation

![Semantic Segmentation screenshot](assets/semantic_segmentation.gif)

As a part of Udacity's Self Driving car nanodegree, I got to spend times on Semantic Segmentation using FCN-8s architecture from the paper name [Fully Convolutional Networks for Semantic Segmentation]

# What is FCN?
  - A Fully Convolution Networks with no Fully Connected Layers.
  - Able to answer where is the object in the image by pixels-wise
  - Since it uses the Fully Convolutional Layers, there is no restriction on the size of the input

## How to convert to FCN
 1. replace fully connected layers with 1x1xn conv layers where n is a number of classes
    - The way Fully connected layers do is to do matrix multiplication over a sliding windows
    - The way FCN do is to run 1x1xn over an input which generate n-dim outputs, and it allows preserving the spatial information.
 2. upsampling the convolutional layers with traposed convolution (Deconvolution?)
    - it helps upsampling the input to the higher order dimension
    - `kernel_size` is used to determine the scope we would like to get the information from
    - `stride` is used to determined the output shape 
 3. Skip Connection - use the information from the previous layers, it allows the network to use information from different resolution scales. It leads to more precise Segmentation
    - When we do normal Convolutional layers, the kerner size is smaller than the original image, and it might contains only some parts of the image which might loses the bigger picture as a result. Therefore, to preserve the information, the skipping connection can be applied from the output of a layer to the non-adjacent layers, so that the far away layer can retain those information.

Okay, enough for the theory, but how to make use of this load of information?

## FCNs in real life
  - take a pre-trained model such as VGG16
  - cut out the FC layer
  - use the 1x1 convolutional layer instead
  - do the transposed convolution
  - add the skip connection, but don't add too much of it because it will cause the model to be really large

That's about it :D

The code of this project can be found in this repo :). Note that it is tested with Tensorflow 1.3.0 only.

[Fully Convolutional Networks for Semantic Segmentation]: https://arxiv.org/abs/1411.4038 

### Setup
##### Frameworks and Packages
Make sure you have the following is installed:
 - [Python 3](https://www.python.org/)
 - [TensorFlow](https://www.tensorflow.org/)
 - [NumPy](http://www.numpy.org/)
 - [SciPy](https://www.scipy.org/)
##### Dataset
Download the [Kitti Road dataset](http://www.cvlibs.net/datasets/kitti/eval_road.php) from [here](http://www.cvlibs.net/download.php?file=data_road.zip).  Extract the dataset in the `data` folder.  This will create the folder `data_road` with all the training a test images.
