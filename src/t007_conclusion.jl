# # Tutorial 7: Putting It All Together
# The purpose of this library is to cover the remaining topics which do not easily fit within another previous topic. Namely, this includes going over auto-differentiation in detail, going over GPU memory stuff, and talking about types of models that can be used for Flux.
# In this tutorial we will cover the following
#  - GPU computation, requires CUDA to be installed
#  - CUDA installation instructions, if needed
#  - GPU movement, storage, and computation via the gpu function. [Example here](https://github.com/FluxML/Flux.jl/blob/645aa044644e36b34686286fcd12949c31a3ab68/test/cuda/cuda.jl)
#  - [CuArrays Library](https://github.com/JuliaGPU/CuArrays.jl)
#
#
# # Setup
# <load all the libraries here>

using Flux

# # First Example
# for the next examples
x = 1
print(x)
