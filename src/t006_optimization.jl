# # Tutorial 6: Flux Library: Optimization in Flux.jl
# The purpose of this chapter is to take a deeper look at how optimization in Flux works. Basically, we’ll dig deeper into Flux.train object, and try to understand the relationship between the three critical components data, model, and optimization routine.
# In this tutorial we will cover the following
#  - [Flux.train! Function](https://github.com/FluxML/Flux.jl/blob/master/src/optimise/train.jl)
#  - Flux.train arguments: loss function, model parameters, and call back
#  - Inspection of each Flux.Train arguments
#  - A deep dive into Flux.train is “learning” models by calling an optimizer
#
# # Setup
# <load all the libraries here>

using Flux

# # First Example
# for the next examples
x = 1
print(x)
