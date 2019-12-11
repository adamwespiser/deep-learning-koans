# # A Demo Koan
# array indexing in Julia is 1-based
xarray = ["a", "b", "c", "solution"]
ind = 0 # Fix me !
ind = 4 #src
@assert xarray[ind] == "solution"
