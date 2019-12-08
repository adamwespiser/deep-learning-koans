# # Tutorial 2: Working with data in Julia
#
# What's covered:
#  - Multi-dimsional [Array assignment](https://docs.julialang.org/en/v1/base/arrays/#Indexing-and-assignment-1)
#  - [Cartesian Index](https://docs.julialang.org/en/v1/base/arrays/#Base.IteratorsMD.CartesianIndices)
#  - Dot Operators
#  - Generating matrixes of zeros, ones, diaganols
#  - Random numbers, seeds with Random.seed!, and generating random matrixes
#  - [Reading in a file, line by line into a string](https://en.wikibooks.org/wiki/Introducing_Julia/Working_with_text_files)
#  - Using BSON.jl for serialization: read/write, talk about JuliaIO

# ## Library Imports
# We need test, for access to `@test`and Random.
module T001 #src
using Test
using Random
@show a
# # Multi-dimsional Assignment
# For multideminsional arrays, we have to do assignment within brackets
# Let x be a multidemisional array of zeros, set the first row equal to ones
x = zeros(3,3)
y = zeros(3,3)
y[1,1] = 1
y[2,1] = 1
y[3,1] = 1
x[:,1] = ones(3) #src
@test x == y

# # Cartesian Indexies
# Sometimes we want to be able to pass around a multidimensional array Index
ind = CartesianIndex()
ind = CartesianIndex(2,2,2,1) #src
#= Change the arguments of Cartesian Index so ind variable is correct =#
marray = reshape(1:16, (2,2,2,2))
@test marray[ind] == 8

# # Dot Operators
# When working with a function in julia, we can apply it element-wise to a an array by
# using the dot operator. For instance, a vector x, in the experssion "x .* 3" would be
# element-wise times 3.
# Using the function `add_eleven_to_arg`, apply it element-wise to our multi-dimensional array, `marray`

function add_eleven_to_arg(x)
  return x + 11
end
x = zeros(5)
#= set xtransformed as a function of x and add_eleven_to_arg =#
xtransformed = x # Modify me !
xtransformed = add_eleven_to_arg.(x) #src
@test xtransformed == repeat([11],5)

# # Initialize Arrays w/ Zero, Ones
# There are a few standard ways to initialize Arrays.
# The basic form is `fn(TYPE, DIM)` whre fn is one of `zeros`, `ones`, and equivelently, `fill`.
# use `fill` to get the following functions to work
mat_ones = ones() # change this to a call to fill
mat_ones = fill(1, (10,10,10)) #src
mat_zeros = zeros() # change this to a fill call
mat_zeros = fill(0, (10, 10, 10)) #src
@test ones(10,10,10) == mat_ones
@test zeros(10, 10, 10) == mat_zeros


# # Random
# Julia gives you a few tools for manipulating random numbers.
myseed = 1 #= set the myseed equal to something else, besides 1 =#
myseed = 2 #src
Random.seed!(myseed)
@test rand(1:10^20) == 6712802529292398594

# # Random: Explicitly setting the set
# Given that we know the seed is 2, call rand w/ with range 1:10^20
# without using the Random.seed! call
myseed = 2
#= set random_num below =#
random_num = rand(1:10^20)
random_num = rand(MersenneTwister(myseed),1:10^20) #src
@test random_num == 6712802529292398594

# # Read a file in line by line
# in julia we can read a file in, line by line
cd("./test") #src
in_file = "../src/assets/t002_data_ingestion/basic.txt"
file_contents = in_file
#= read in_file into file_contents =#
file_contents = read(in_file, String) #src
@test file_contents == "apple\nbanana\norange\npear\ngrapes\n"
cd("..") #src

# # Using BSON
# imports.
# WIP
using BSON
dir = "../src/assets/t002_data_ingestion/"
@test true


@show "t002 Done" #src
#= end module =#
end #src
