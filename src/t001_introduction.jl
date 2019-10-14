# # Tutorial 1: Introduction to Julia and Koans
#
# In this tutorial we will cover the following
#  - Julia basics
#  - What is a Koan
#  - What is Flux
#
# Problem Statement: In our first tutorial, we want to get used to the concent of using IPython Notebeoks
# with Julia to solve Koans, and gain some exposure to programming concepts in Julia, and well as introduce
# Flux.jl from a high level
#
# Look! Images that are in the assets directory can be used here!!!
# ![](../src/assets/t001_introduction/fig0.png)
#
# # Setup
# We will need to load in Flux
#
using Flux

# # A first Koan
# Create a function, `f`, such that our assert function passes
function f(arg) # inline code comments
  return arg #src
end
x = 1
@assert [f(x) for x in 1:10] == 1:10

# # A Second Koan
# Create a stuct, `s`, that will satisfy the following conditions:
# - Struct S has a field x
# - Struct S has an inner constructor that accepts no arguments and sets x to 42
struct S
  x :: Int64    #src
  S() = new(42) #src
end
@assert S().x == 42
