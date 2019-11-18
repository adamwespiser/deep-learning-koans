# # Tutorial 4: Flux Library: Convolutional Networks, Example Layers
#
# In this tutorial we will cover the following
#  - What is a Convolutional Neural Network (See Hinton 2015)
#  - Data sources for CNN Networks, using [MLDatasets](https://github.com/JuliaML/MLDatasets.jl) to load MINST
#  - A look at [Convolution Layers, and their arguments](https://github.com/FluxML/Flux.jl/blob/master/src/layers/conv.jl#L35)
#  - ConvLayers: adding, stride, width/height, activation, Determining what padding=”same” would be
#  - Convolutional Layers as dimensional manipulation
#  - [Inverse Convolutional Networks](https://github.com/FluxML/Flux.jl/blob/master/src/layers/conv.jl#L71), possibly with VAE
#  - Putting it all together with [MINST Classifier](https://github.com/FluxML/model-zoo/blob/master/vision/mnist/conv.jl) and [VAE w/ Conv/Deconv](https://github.com/FluxML/model-zoo/blob/master/vision/mnist/vae.jl)
#
# # Load Flux
# We are just going Flux
using Flux
using MLDatasets


# # What is a Convolutional Neural Network (See Hinton 2015)
# A convolutional neural network is a network that uses convolutional transformations
# as part of the neural network, and are used for both classification and generative tasks.
# The recent revolution in Convolutional Neural Networks was spawned in the Hinton lab, by applying
# a CNN to outperfom alternative approaches on ImageNet.
# [Krizhevsky 2012 paper on ImageNet (from Hinton's Lab)](https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf)

# # Data Sources for Convolution Neural Networks
# We are going to use ImageNet, which is available from MLDatasets
test_x, test_y = MNIST.testdata()
train_x, train_y = MNIST.traindata()

train_x_size = (0,) # fix me!
train_x_size = (28,28, 60000) #src
train_y_size = (0,) # fix me !
train_y_size = (60000,) #src
@assert size(train_x) == train_x_size
@assert size(train_y) = train_y_size


# # Reshaping to WHCN Format
# For our MINST dataset, we will want to work with image data that fits
# in the following, standardized dimensions (W, H, C, N).
# where W is width in pixels, H is width in Pixels, C is number of channels, and N is number of samples.
# The Type of the array should be Array{Float, 4}
# This koan asks you to sample 100 images from MINST and convert then to WHCN format.
N = 100
x_train_whcn = train_x
X = reshape(float.(train_x[:,:,1:N]), 28, 28, 1, N)  #src
y = train_y
y = float.(train_y[1:N]) #src
@assert size(X) == (28,28,1,100)
@assert size(y) == (100,)


# # Finding the output dimensions of a Conv Layers
# Given the convolution, find the output dimension after applying it to MINST
layer = Conv((1, 1), 1 => 32, relu, stride = (1, 1))

output_dims = (0,) # set this
output_dims = (28, 28, 32, 100) #src
@assert size(layer(X)) == output_dims

# # Find the Convolution that fits a shape
# We want to transform our 100 images of MINST such that
# our W/H is 27, and our number of channels is 42. What is a layer that will acccomplish this?
layer = Conv((1,1), 1=>1, identity)
layer =  Conv((2, 2), 1 => 42, identity, stride = (1, 1)) #src
@assert size(layer(X)) == (27, 27, 42, size(X,4))

# note, we introduce the size(X,4) motif here instead of hardcoding the number of
# of examples. Further, the activation function we are using identity, has the property
@assert identity.(X) == X
@assert identity.(y) == y

# # Stride
# Stride, or how many pixels height/width are skipped per convolutional filter
# step, is one way to manipulate how the filter is subsequently applied to each
# image. Taking the following layer, modify its stride to get it to pass the
# dimension of the output.
layer = Conv((1, 1), 1 => 32, relu, stride = (1, 1))
layer = Conv((1, 1), 1 => 32, relu, stride = (2, 1)) #src
@assert size(layer(X)) == (14, 28, 32, size(X,4))
