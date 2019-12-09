
# # Tutorial 6: Flux Library: Optimization in Flux.jl
# The purpose of this chapter is to take a deeper look at how optimization in Flux works. Basically, we’ll dig deeper into Flux.train object, and try to understand the relationship between the three critical components data, model, and optimization routine.
# In this tutorial we will cover the following
#  - [Flux.train! Function](https://github.com/FluxML/Flux.jl/blob/master/src/optimise/train.jl)
#  - Flux.train arguments: loss function, model parameters, and call back
#  - Inspection of each Flux.Train arguments
#  - A deep dive into Flux.train is “learning” models by calling an optimizer
#
# The source code for Flux's optimization code is [available here](https://github.com/FluxML/Flux.jl/tree/master/src/optimise)
#
# # Setup
# Load Flux
module T007 #src
using Flux
using Random
using Test
# ## Set up some basic variables
# Note: You may need to re-initialize these variables for
# Flux.train to show a difference in parameters needed
# to pass some of the tests
Random.seed!(42);
function create_model()
  Chain(
    Dense(784, 32, σ),
    Dense(32, 10), softmax)
end

m = create_model()
loss(x, y) = Flux.mse(m(x), y)
ps = Flux.params(m)

opt = ADAM(0.1)
x = rand(784);
y = rand(10);
data = Iterators.repeated((x, y), 50);


# #### Randomize model
# This is a helper function that re-initializes the model,
# allowing us to learn it many times with different staring states.
# Alternatively, we have `reinitializie_model`, which will reset
# the m to the original params.
# Prove that two `create_model` calls do not create the same
# set of parameter

m1 = randn(1) # Fix me !
m1 = params(create_model()) #src
m2 = m1 # Fix me !
m2 = params(create_model()) #src
@test m1 != m2


# ## More on Flux Train
# [Flux.Train] is the core of Flux's Machine Learning model
# and is responsible for updating the model parameters, via
# an loss function, to the data seen by the model
# The basic train function is as follows:
m = create_model()
ps_old = params(m)
#= In this line, make a call to Flux.train! =#
Flux.train!(loss, params, data, opt) #src
@test ps_old != ps && ps_old == ps_old

# Thus, we can see that the parameters of the model are updated!

# ## Flux train callback functions
# Create a lambda function that accepts no input, and simply returns 0.
lambda_function = identity # Fix me!
lambda_function = () -> 0.0 #src
@test [lambda_function() for i in 1:10] ==  zeros(10)

# ## Flux.train callbacks
# In flux, we can make a call back that runs an effect, in our case,
# we will want to print out a message to IO. Write a function that
cb = identity # Fix me
cb = () -> println("Hello World!")
@test typeof(cb()) == Nothing

# ## Flux.train callbacks
# Now, we can write a useful callback function that will test our loss
# for every iteration of execution
test_x, test_y = x, y
evalcb = () -> println("some function to report loss")
evalcb = () -> @show(loss(test_x, test_y)) #src
Flux.train!(loss, ps, data, opt, cb = evalcb)
@test true || "Successfully got here"

# ## Callback thresholding/throttling
# Flux gives us the ability to throttle the number of times we
# invoke our callback function. This function is called
# `throttle` and it accepts a callback, and an Integer n, which
# prevent the callback from being called more than n times per second
# [Throttle Source Code Here](https://github.com/FluxML/Flux.jl/blob/master/src/utils.jl#L115)
m = create_model()
test_x, test_y = x, y
evalcbt = () -> @show(loss(test_x, test_y)) # Wrap this in a call to throttle
evalcbt = Flux.throttle(evalcb, 10) #src
Flux.train!(loss, ps, data, opt, cb = evalcbt)
@test true || "Successfully got here"


@show "t006 Done" #src
#= end module =#
end #src
