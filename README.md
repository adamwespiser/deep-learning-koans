# Deep Learning Koans
A collection of Deep Learning Koans written for Flux.jl and Julia Programming language.

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
