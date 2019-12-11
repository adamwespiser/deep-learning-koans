# Deep Learning Koans
A collection of Deep Learning Koans written for Flux.jl and Julia Programming language.

#### Colab Link
[Run these koans on Colab](https://colab.research.google.com/github/adamwespiser/deep-learning-koans/blob/master)



### What is a Koan?
A Koan is a short puzzle designed to elucidate understanding upon completion. The plural, "koans" are also a format of interactive programming puzzles designed to be run locally and to teach the user programming, through a series of challenges in order of increasing difficultly. This is exactly what this project is!    
Thus, a koan critically consists of three components:     
 - A text introduction to a concept and problem
 - Section of code that applies the concept
 - A test that is not passed untill the concept is applied correctly

This format borrows a lot from the pedalogical approach of Daniel P. Friedman, and his book, [The Little Schemer](https://www.ccs.neu.edu/home/matthias/BTLS/).    
A list of koans projects for a variety of platforms and languages [can be found here on github](https://github.com/ahmdrefat/awesome-koans/blob/master/koans-en.md).    


### So what's this project trying to do?
This project attempts to take the Koans approach to teaching programming, and apply that to the subject matter of deep learning in Julia.  
To do this, we will focus first on giving a skippable overview of Julia, then move on to demonstrating the library [Flux.jl](https://github.com/FluxML/Flux.jl).  
Technically, this project is implemented as a series of IJulia Notebooks that you generate from source code, host locally with IJulia, the extension to run Julia in a Jupyter Notebook, then interactively modify the code until "it works". 
Links and additional resources are scattered throughout the material.    

### Why Julia?
First, Julia offer a couple advantages compared to Python and R that make it superior for some projects, namely, its gradual typing improves the developer experience and code reliability, and compilation to LLVM makes Julia code faster than R/Matlab/Python. 
However, Julia is young, and does not have the depth and breadth of machine learning, and web development packages that make Python a safe production choice, or the depth of statistical packages and academic work that makes R so revered.     
Nonetheless, Julia's LLVM compilcation offers huge advantage for Deep Learning in research and practice: we can actually program using a single language, instead of having to call an underlying C++ library, like Tensorflow or PyTorch. This requires a programming language, differential programming, which is currently being studied and integrated into Julia through the Zygote project

### Why Flux?
Flux is a Julia only, neural network library, with basic differential programing capability, with a library of optimizers, layers, and helper functions that can facilitate deep learning, and has first-class support for GPU allocation.
Thus, solutions like Julia/Flux, that use a single language to compile to fast LLVM bytecode and leverage GPUs, are a good future for for Deep Learning, as they are conceptually simpler to understand than current solutions like PyTorch/TensorFlow with the same performance. However, and you'll see this very soon throgh the koans, the API maturity is not quite there! 


# Chapters/Contents
The chapter outline is as follows:
 - 1: [Introduction to Julia](notebooks/t001_introduction.ipynb)
 - 2: [Working With Data In Julia](notebooks/t002_data_ingestion.ipynb)
 - 3: [Intro to Flux](notebooks/t003_flux_intro.ipynb)
 - 4: [Convolutional Neural Networks and Layers in Flux](notebooks/t004_conv_layers.ipynb)
 - 5: [Recurrent Neural Networks and Layers in Flux](notebooks/t005_recurrent_layers)
 - 6: [Flux optimization](notebooks/t006_optimization.ipynb)
 - 7: [Putting in all together, and more examples!](notebooks/t007_conclusion.ipynb)

#### Attribution Note
Many examples of these koans are borrowed and modified from sources in the Flux source code, or examples in the Flux Documentation. 
The Flux source code can be found here: [Flux.jl](https://github.com/FluxML/Flux.jl).


# How to Run
To run this project, you will need Julia installed locally, [which is available here](https://julialang.org/downloads/index.html).  Additionally, you will need NVIDIA drivers installed on your machine, for the GPU based examples in Chapter 7.    
Clone the Repository, or download directly from Github.    
```
git clone  https://github.com/adamwespiser/deep-learning-koans.git
```

Run julia in the `deep-learning-koans` directory    
```
$ cd deep-learning-koans/
$ julia --project='.'
```

Instantiate the environment, which will download all the needed packages    
```
julia> using Pkg; Pkg.instantiate();
```

Build the notebooks    
```
julia> include("deps/build.jl")
```

Open the Julia Notebooks    
```
julia> using IJulia
julia> notebook(dir=".")    
```

# Dev
The projects is built using the script found in [`deps/build.jl`](deps/build.jl). 

#### Adding a package to Project.toml, Manifest.toml
For pkg X, we can get the UUID with the following:    
```
julia> Pkg.METADATA_compatible_uuid("X")    
```
Then, add an entry to `[deps]` in Project.toml.    
Finally, we will want to re-populate the Manifest.toml, which we can do as followings:    
```
$ julia --project="."    
julia> Pkg.resolve()    
julia> Pkg.instantiate()    
```

#### Test
Run all the notebooks at once.    
```
$ julia --project="." test/run_koan_source.jl
```

