using Literate

repo_src = joinpath(@__DIR__, "..", "src")
notebooks_dir = joinpath(@__DIR__, "..", "notebooks")

files = [
  "t001_introduction" => "t001_introduction",
  "t002_data_ingestion" => "t002_data_ingestion",
  "t003_flux_intro" => "t003_flux_intro",
  "t004_conv_layers" => "t004_conv_layers",
  "t005_recurrent_layers" => "t005_recurrent_layers",
  "t006_optimization" => "t006_optimization",
  "t007_conclusion" => "t007_conclusion"
]

Sys.rm(notebooks_dir;recursive=true,force=true)
for (file,name) in files
  file_jl = file*".jl"
  Literate.notebook(joinpath(repo_src,file_jl), notebooks_dir; name=name, documenter=false, execute=false)
end
