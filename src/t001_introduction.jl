# # Tutorial 1: Introduction to Julia and Koans
#
# In this tutorial we will cover the following
#  - Julia basics
#  - What is a Koan
#  - Functions
#  - Methods
#  - Abstract Data Types
#  - Arrays
#  - Types
#  - Control flow
#  - Macros

# Problem Statement: In our first tutorial, we want to get used to the concent of using IPython Notebeoks
# with Julia to solve Koans, and gain some exposure to programming concepts in Julia, and well as introduce
# Flux.jl from a high level
#
#
# # Setup
# We will need to load in Flux
#
module T001 #src
using Test
using Flux
include("../src/check_cuda.jl")
print_system_gpu_status()

# # What's a koan?
# A Koan is a paradoxical anecdote or riddle, used in Zen Buddhism to demonstrate the inadequacy of logical reasoning and to provoke enlightenment.
# For us, Koans will be used to teach the concepts of Deep Learning in Julia, and we'll try to keep it all very logical. :)
# To "solve" a koan, you will be required to add a line of code to the cell, which will allow a final
# assertion to pass. For instance:
x = 1
y = 1 #src
@test x == y

# This can be solved in ONE conventional way:
# - By defining a new variable `y = 1`
#
# Thus, for the sake of these Koans, we suppose the @test statement does not need to be modified
# in order to solve the problem!
# Basically anything goes, as long as `@test x == ...` is not modified into `@test true`



# # A 0th Koan
# This Koan serves only to introduce you to Koans, and the directory structure of the project.
# To Solve this Koan, find this notebook definition in the source code, and figure out what
# we should set `x` to be to pass the assertion
using Random
aboutkoan = "set the myseed equal to something else, besides 1"
myseed = 1
myseed = 42 #src
Random.seed!(myseed)
@test rand(1:10^20) == 5590813852184710016

# Thus, by looking in "src/t001_introduction.jl", you can find the defition of the function in markup.
# Missing lines will have a "#src" comment. These usually contain the information needed to complete
# the koan.
# If you are ever stuck, you can always look up the markup defition, and find a working solution to
# get the assertion to pass!


# # A Function Koan
# Create a function, `f`, such that our assert function passes
# and f is a unary, or single argument function.
# More on functions can be found within the [Julia Docs](https://docs.julialang.org/en/v1/manual/functions/#man-functions-1).
function f(arg) # modify the next line to return the argument...
  return arg #src
end
x = 1
@test [f(x) for x in 1:10] == 1:10

# # An Overloaded Function Koan
# Starting with `f` which accepts a String and returns an `Int64`, make two modifications:
# - Overload  `f` to accept an `Int64` and returns 0 of type `Int64`
# - Overload  `f` so it accepts a `Float64` and return 1 of type `Int64`
#
# Thus `f` is known as a "Method". [Julia Docs](https://docs.julialang.org/en/v1/manual/methods/)

x_float = convert(Float64, 1)
x_int = convert(Int64, 1)

f(x::String)::Int64 = 10
f(x::Int64)::Int64 = 1    #src
f(x::Float64)::Int64 = 0  #src
@test f(x_int) == f(x_float) + 1 && x_int == f(x_int)

# Notice that running this cell without modification returns a `MethodError: no method matching ...`
# and then the type signature of the function at the invocation. When you see a MethodError, it's
# a huge clue that you are

# # A Struct Koan
# Structs are just simple data containers in Julia. [Docs](https://docs.julialang.org/en/v1/base/base/#struct)
# Create a simple struct Point, with two accessors, x and y
struct Point
  x
  y
end
@test Point(1,1).x == Point(1,1).y

# # A Mutable Struct
# Taking our point struct from the previous koan, get the following to work
# by re-writing Point as a mutable struct `PointMutable`
struct Point
  x
  y
end

mutable struct PointMutable #src
  x #src
  y #src
end #src

p1 = Point(0,0)
p1 = PointMutable(0,0) #src
p2 = Point(1,1)
p2 = PointMutable(1,1) #src
setfield!(p2, :x, getfield(p1, :x))
setfield!(p2, :y, getfield(p1, :y))
@test p1.x == p2.x


# Create a stuct, `s`, that will satisfy the following conditions:
# - Struct S has a field x
# - Struct S has an inner constructor that accepts no arguments and sets x to 42.
# [Julia Docs on Inner Constructor Methods](https://docs.julialang.org/en/v1/manual/constructors/#Inner-Constructor-Methods-1)
struct S
  x :: Int64    #src
  S() = new(42) #src
end
@test S().x == 42

# Note, this might seem a little esoteric, but this technique is a very common way to initialize a struct to default values.


# # Array Koans
# [Julia Docs: Array Page](https://docs.julialang.org/en/v1/base/arrays/)
# We can create random arrays using the `zeros` or `ones` function
# Use either one of these to create a 10x10x10 matrix, and get its shape or `size`
mat = rand(10) # change the call to rand, (_add more args)
mat = rand(10,10,10) #src
@test size(mat) == (10, 10, 10)


# # Initialize Arrays
# There are a few standard ways to initialize Arrays.
# The basic form is `fn(TYPE, DIM)` whre fn is one of `zeros`, `ones`, and equivelently, `fill`.
# use `fill` to get the following functions to work
mat_ones = ones() # change this to a call to fill
mat_ones = fill(1, (10,10,10)) #src
mat_zeros = zeros() # change this to a fill call
mat_zeros = fill(0, (10, 10, 10)) #src
@test ones(10,10,10) == mat_ones
@test zeros(10, 10, 10) == mat_zeros

# # Array Indexing
# For a 10x10x10 array, get the item at the (4,5,6) position
mat = reshape(1:1000, 10, 10, 10)
mat_value = mat[4,5,6] #src
#= set mat_value to be a value inside mat =#
@test 544 == mat_value

# # Array slicing
# For our same 10x10x10 sequential matrix, we want to take:
# - The first item in the first dimension
# - All the items in the second dimension
# - The fourth item in the third dimension
# Note: Every item in mat has 3 dimensions. This solution will be the collection of items that is the intersection
# between the set of items with their first dimension equal to 1, and their second dimension equal to 4.
about_koan = "set a variable mat_value equal to the slice..."
mat_value = mat[1, :, 4] #src
@test mat_value == [301, 311, 321, 331, 341, 351, 361, 371, 381, 391]

# For more information, see `Base.Colon` in the [Julia Docs](https://docs.julialang.org/en/v1/base/arrays/#Base.Colon)

# # Array CheckBounds
# Using the `checkbounds function` [Julia Docs: Base.checkbounds](https://docs.julialang.org/en/v1/base/arrays/#Base.checkbounds)
# Fix the following function to access the array, or return 0 if the bounds are violated
function wrap_checkbound(M, i, j)
  status = checkbounds(Bool, M, i, j) #src
  if status # define status variable or re-write w/ a call to checkbounds
    return M[i,j] #src
  else
    return 0
  end
end
mat = ones(3,3)
@test wrap_checkbound(mat, 3, 3) == 1
@test wrap_checkbound(mat, 1, 3) == 1
@test wrap_checkbound(mat, 1, 0) == 0
@test wrap_checkbound(mat, 0, 0) == 0
@test wrap_checkbound(mat, 0, 1) == 0

# Hopefully this helps you remember that arrays are 1-indexed, but if you forget, you can always
# write your code to use check_bound and not worry about it!

# # Array Reshaping
# Say we have an image library of 28 height, 28 width Black and White pixel images, and we have 400 of them.
# Use the function, resahpe, to transform this into WHCN format, which stands for (Width) (Height) (Channel)
# (Number of Examples). For us, we want 1 channel.

old_mat = rand(28,28, 400)
new_mat = reshape(old_mat, 28, 28, 400) # CHANGE THIS LINE
new_mat = reshape(old_mat, 28, 28, 1, 400) #src
@test size(new_mat) == (28, 28, 1, 400)


# # Types
# Type level programming is a big part of Julia, and the [Docs](https://docs.julialang.org/en/v1/manual/types/) are a great place for in-depth information.
# To start, useful function in Julia to inspect types at runtime is the 'typeof' functions.
# Since we just covered Arrays, lets start off by using the "typeof" function to
# satisfy three assertions.

mat_single = zeros(0)
mat_single = Array{Float64,1}() #src
@test typeof(mat_single) == Array{Float64,1}

# Thus, we can take the results of typeof, `Array{T, N}` and call this as a function to initialize
# an array, without any values, if need be. Types are a big subject in Julia, but there are a couple
# things you should know:
# - Types are dynamic, they are runtime, not compile-time objects, and they can change over time for a symbol
# - Julia uses "Gradual Typing" which is a technical term for saying, "you can skip type annotations if you don't know or feel like adding them"
# - Julia allows both concrete types you can work with, and abstract types, which can be used to parameterize other functions and types.
#
# For practical programming, types give us a few advantages
# - They speed up run time, since we can compile away dynamic dispatch
# - They make code "safer" preventing us from misusing functions, and giving us better error messages when we call a function with incorrect arguments.
#
# However, for these advantages, we add considerable complexity, and probably the nicest feature of Julia's type system is that we don't have to worry about types if we
# really dont' have the overhead!


# # Abstract Types
# Abstract Types cannot be instantiated, but they can be used for dispatching functions and pattern
# matching arguments.
# The basic form of an Abstract Type "Is parent of" relationship is: "T <: U" which says:
# - U is a Parent Type of T
# - and "U <: T" evaluetes, as always during runtime, to a boolean value!
abstract type MyCustomType end
IntegerParent = MyCustomType # replace MyCustomType with something else
IntegerParent = Number #src
Float64Parent = MyCustomType # replace MyCustomType with something else
Float64Parent = AbstractFloat #src

@test Integer <: IntegerParent
@test Float64 <: Float64Parent

# Note: These are kind of tricky without looking in the [Julia Docs on Abstract Types](https://docs.julialang.org/en/v1/manual/types/)
# which will explain the numerical hierarchy.


# # Control flow / ifelse
# First, we'll start with ifelse, which accepts a boolean and returns
# the first argument if the boolean in true, else returns the second
# Use if else to map [true, false, false] onto [1, "banana", "banana"]
bool_vec = [true, false, false]
target = [1, "banana", "banana"]
fn(bools) = bools # REWRITE THIS FUNCTION using ifelse
fn(bools) = map(x -> ifelse(x, 1, "banana"), bools) #src
@test fn(bool_vec) == target

# Additionally, we notice the type of "target" is `Array{Any, 1}`, which is a hetergeneous array.

# # Control Flow / if else elseif end
# Using an enum defintion provided, use an if/elseif/else to match the first, second, and rest
# of enums, in order, returning -1, 0, 1, for (e1, e2, e3)
@enum BasicEnum begin
  e1
  e2
  e3
end

about_koan = "write this function, matching on arg"
function match_enum(arg::BasicEnum)
  if arg == e1      #src
    return -1
  elseif arg == e2  #src
    return 0        #src
  else              #src
    return 1        #src
  end               #src
end

@test match_enum(e1) == -1
@test match_enum(e2) == 0
@test match_enum(e3) == 1


# # Control Flow / for loop
# Julia has for loops compile to be just as fast as vectorized code,
# Thus we can write some code to sum all the numbers in a matrix using a for loop.
#= a basic for loop would be =#
vec = 1:10;
mutable struct Count
  x :: Int64
end
c = Count(0) # add a for loop below to iterate over vec and add each element to c
for i = eachindex(vec) #src
  @show vec[i]         #src
  c.x += vec[i]        #src
end                    #src
@test c.x == Count(55).x

@show "t001 Done" #src
#= end module =#
end #src
