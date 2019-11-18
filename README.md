# Deep Learning Koans
A collection of Deep Learning Koans written for Flux.jl and Julia Programming language.

### What is a Koan?
A Koan is a short puzzle designed to elucidate understanding upon completion. The plural, "koans" are also a format of interactive programming puzzles designed to be run locally and to teach the user programming, through a series of challenges in order of increasing difficultly. This is exactly what this project is!    
This format borrows a lot from the pedalogical approach of Daniel P. Friedman, and his book, [The Little Schemer](https://www.ccs.neu.edu/home/matthias/BTLS/).    
A list of koans projects for a variety of platforms and languages [can be found here on github](https://github.com/ahmdrefat/awesome-koans/blob/master/koans-en.md).    


### So what's this project trying to do?
This project attempts to take the Koans approach to teaching programming, and apply that to the subject matter of deep learning in Julia.  
To do this, we will focus first on giving a skippable overview of Julia, then move on to demonstrating the library [Flux.jl](https://github.com/FluxML/Flux.jl).  
Technically, this project is implemented as a series of IJulia Notebooks that you generate from source code, host locally with IJulia, then interactively modify the code until "it works". 
Links and additional resources are scattered throughout the material.    


# Chapters/Contents
So far, the tentative chapter outline is as follows:
 - Introduction to Julia
 - Working With Data In Julia
 - Intro to Flux
 - Convolutional Neural Networks and Layers in Flux
 - Recurrent Neural Networks and Layers in Flux
 - Putting in all together, and more examples!


# How to Run
Clone the Repository    
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
The projects is built using the script found in `deps/build.jl`. 

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


