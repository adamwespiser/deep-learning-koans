# # Tutorial 3: Intro To Flux
#
# [Flux White Paper](https://arxiv.org/pdf/1811.01457.pdf)
#
# In this tutorial we will cover the following
#  - About Flux, [here](https://github.com/FluxML/Flux.jl)
#  - What is a Flux model, [ModelZoo](https://github.com/FluxML/model-zoo), models as functions, Flux model types
#  - Flux Layer Introduction
#  - Flux Optimizer Introduction
#  - Optimization batches, step size, batch normalization concepts
#  - Different optimizer types, SGD, Flux.ADAM, et cetera
#  - Flux activation functions: tanh, relu, leaky-relu
#


using Flux

# # Set up the data
# for the X and y, we want to take X and subsequently predict y

X_0 = randn(100,2)
X_1 = randn(100,2) .+ 1
X = transpose(cat(X_1, X_0, dims = 1))
y = cat(ones(100), zeros(100), dims = 1)

# This will be the data that we use, it's just a simple X/y
# dataset of N = 200, two dimensions, and one output dimension, y


# # Basic Model
# We will set up a basic neural network, where we will take the two
# dimension per data point, expand it out to three, then contract, and
# apply softmax.
m = Chain(
  Dense(2,3),
  Dense(3,1,σ),
  x -> reshape(x, :,1)
)

# # Model Intuition Test
# Given our model, `m`, and our data, `X`, we can apply our data to the
# model as if it where a function. What are the dimensions of `m(X)`?
n = 42
Xp = randn(2,n)
y_predicted = m(Xp)
about_koan = "set y_predicted_size w/ n"
y_predicted_size = (0,0)
y_predicted_size = (2, n) #src
@assert size(y_predicted) == y_predicted_size # some function of n !


# # Flux optimizer
#
loss(X, y) = Flux.mse(m(X), y)
ps = Flux.params(m)
opt = Flux.ADAM()

Flux.train!(loss, ps, [(X,y)], opt)

# Note, it can be tricky to get the arguments to Flux.train correct
# what we are looking for, is that for each subsequent round, the (X,y)
# pair sufficiently matches the signature used in the loss function.
#
# Using the following loss function, rewrite the Flux.train function
loss(X) = Flux.mse(m(X), y)
ps = Flux.params(m)
opt = Flux.ADAM()
data = [(X,y)]
about_koan = "set data variable"
data =  [(X,)]
Flux.train!(loss, ps, data, opt)

# using the loss function this way is a little cleaner, as we won't
# have to pass along y for every batch

# # Batch Sizes
# thus, using the previous formulation of our `data` and `loss` function
# into Flux.train, we batch by passing in a list of inputs that fit the dimensions
# of the model

ps = Flux.params(m)
opt = Flux.ADAM()
N = 200
X = randn(MersenneTwister(42), 200, 2)
y = cat(ones(100), zeros(100), dims = 1)

N_total = size(X, 1)
batch_size = 20
batch_idx = collect(Base.Iterators.partition(1:N_total, batch_size))
X_batches = zip([transpose(X[1:batch_size,:])],[ y[1:batch_size]] )
X_batches = zip([transpose(X[i,:]) for i in batch_idx], [y[i] for i in batch_idx]) #src
Flux.train!(loss, ps, X_batches, opt)


# # Different Optimizers
# there are a number of different optimizers you can try, for instnace,
# you can you a simple Descent optimizer [more info here](https://pkg.julialang.org/docs/Flux/QdkVy/0.8.3/training/optimisers/)
using Flux, Flux.Tracker

W = param(rand(MersenneTwister(42),2, 5))
b = param(rand(MersenneTwister(42),2))

predict(x) = W*x .+ b
loss(x, y) = sum((predict(x) .- y).^2)

x, y = rand(MersenneTwister(42),5), rand(MersenneTwister(42),2) # Dummy data
l = loss(x, y) # ~ 3

θ = Params([W, b])
grads = Tracker.gradient(() -> loss(x, y), θ)

# # Re-write the following example with Flux.Descent
using Flux.Tracker: grad, update!

η = 0.1 # Learning Rate
for p in (W, b)
  Flux.Tracker.update!(p, -η * grads[p])
end
@assert W[1,1] == 0.3967483862667399


# # Re-written example w/ Flux.Descent
# Using the function Flux.Descent, which takes as an argument the learning rate, η
η = 0.1 # Learning Rate
opt = "some function of η"
opt = Flux.Descent(0.1) #src

W = param(rand(MersenneTwister(42),2, 5))
b = param(rand(MersenneTwister(42),2))

predict(x) = W*x .+ b
loss(x, y) = sum((predict(x) .- y).^2)

x, y = rand(MersenneTwister(42),5), rand(MersenneTwister(42),2) # Dummy data
l = loss(x, y) # ~ 3

θ = Params([W, b])
grads = Tracker.gradient(() -> loss(x, y), θ)

for p in (W, b)
  # Flux.Track.update!(...) use opt here...
  Flux.Tracker.update!(opt, p, grads[p]) #src
end
@assert W[1,2] == -0.0984933499933695

