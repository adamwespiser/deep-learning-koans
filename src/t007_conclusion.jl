# # Tutorial 7: Putting It All Together
# The purpose of this library is to cover the remaining topics which do not easily fit within another previous topic. Namely, this includes going over auto-differentiation in detail, going over GPU memory stuff, and talking about types of models that can be used for Flux.
# In this tutorial we will cover the following
#  - GPU computation, requires CUDA to be installed
#  - CUDA installation instructions, if needed
#  - GPU movement, storage, and computation via the gpu function. [Example here](https://github.com/FluxML/Flux.jl/blob/645aa044644e36b34686286fcd12949c31a3ab68/test/cuda/cuda.jl)
#  - [CuArrays Library](https://github.com/JuliaGPU/CuArrays.jl)
#
COLAB
# # Setup
# We will be loading in flux!
module T007 #src
using Flux
using Flux: throttle
using Random
using Test
include("../src/check_cuda.jl")
print_system_gpu_status()


# # GPU Examples
# for the next examples, you will need a working a CUDA enabled gpu
# without one, the needed libraries will not be able to build, and
# the koans will pass by default.
# You will need an NVidia graphics card to install CUDA, and unfortunately
# there are only a few macbooks that have NVideo cards (most run Radeon cards)
# However, you have a couple of options, whose exact instructions will be left
# to the reader for the sake of brevity:
#  - [Google Compute Engine](https://cloud.google.com/compute/docs/gpus/) is my personal choice for renting GPU hours. You will have to set up IJulia to work over the net, but this is not too bad.
#  - Run this notebook on [Google Colab](https://colab.research.google.com/drive/1zAkadxc_iJE_oj-TOdwHpfLU2oaNHJOS#scrollTo=cEOANYIVIdR5)
#  - Run this locally on a computer with a GPU
# Of all the potential solutions, I would strongly recommend finding a computer with an NVIDIA GPU, or using an out of the box solution, like Colab.


# # A GPU Layer koans
# To move a layer over to the GPU, we can use the helper function `Flux.gpu`
# to call it, we will use a pipe, or compose function, `|>`, which applies a
# function to the preceeding argument.
# Thus, we have `|> :: a -> (a -> b) -> b`.
# ### Put that koan in your pipe!
# Using the pipe chain together `layer1` and `layer2`
layer1 = Dense(32,16)
layer2 = Dense(16,8)
data = randn(MersenneTwister(42), Float32, 32)
m = Chain(layer1, layer2)
fn = layer2(layer1(data)) # rewrite this with pipes
fn =  z -> (x -> layer1(x) |> y -> layer2(y))(z) #src
@test fn(data) == Chain(layer1, layer2)(data)
# Note, that our pipe construction is very close to what is going on with
# `Chain`. However, Chain gives us a nice convenience function, and allows
# us to avoid using lambda functions

# # Using gpu with data
# Now, we can use pipes, along with the gpu function and some behind the
# take our data x, and move it over to the gpu
if check_cuda()
  data = randn(MersenneTwister(42), Float32, 10)
  data_gpu = copy(data) # modify me!
  data_gpu = data |> gpu
  @test typeof(data_gpu) == CuArray{Float32,1}
else
  print("CUDA library not detected. If you have a GPU please install CUDA.")
end

# # Using gpu with model layers
# scenes magic to convert the storage of layers from CPU to GPU
if check_cuda()
  m = Chain(Dense(81, 40), softmax)
  pr = params(m)
  # convert m params onto GPU
  mgpu = m # Modify me with a piped call to gpu!
  mgpu = m |> gpu #src
  pr_gpu = params(mgpu)

  @test type(pr_gpu) != type(pr)
else
  print("CUDA library not detected. If you have a GPU please install CUDA.")
end

# # Converting Training Call to GPU
# Take th following call to `Flux.train!`, and fully convert the data used
# in training, and the model to the GPU.
if check_cuda() #src

m = Chain(
    Dense(784, 32, σ),
    Dense(32, 32),
    Dense(32, 32),
    Dense(32, 10),
    softmax
  )
m = m # Modify me
m = m |> gpu #src
x = rand(784, 20) # Modify me
x = rand(784, 20) |> gpu
y = rand(784, 20)
for i in 1:20 # Normalize each
  y[:,i] = y[:,i] ./ sum(y[:,i])
end

y = copy(y) # Modify me
y = y |> gpu #src
data = Iterators.repeated((x, y), 50)
loss(x, y) = Flux.mse(m(x), y)
ps = params(m)

opt = ADAM()
evalcb = () -> @show(loss(x, y)) # Wrap this in a call to throttle
Flux.train!(loss, ps, data, opt, cb = Flux.throttle(evalcb, 10) )
end

#= end module =#
end #src
