using Literate

repo_src = joinpath(@__DIR__, "..", "src")
notebooks_dir = joinpath(@__DIR__, "..", "notebooks")

files = [
  "t001_introduction" => "t001_introduction",
  "t002_data_ingestion" => "t002_data_ingestion",
  "t003_model_layers" => "t003_model_layers"
]
Sys.rm(notebooks_dir;recursive=true,force=true)
for (file,name) in files
  file_jl = file*".jl"
  Literate.notebook(joinpath(repo_src,file_jl), notebooks_dir; name=name, documenter=false, execute=false)
end
