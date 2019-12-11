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

function colab_blank(content)
    content = replace(content, "COLAB" => "")
    return content
end

Sys.rm(notebooks_dir;recursive=true,force=true)
for (file,name) in files
  file_jl = file*".jl"
  Literate.notebook(joinpath(repo_src,file_jl), notebooks_dir; name=name, documenter=false, execute=false, preprocess=colab_blank)
end

###################################################################
# colab notebooks

repo_src = joinpath(@__DIR__, "..", "src")
notebooks_dir = joinpath(@__DIR__, "..", "colab/notebooks")

colab_header = read("./deps/colab_header", String)
function colab_install(content)
    content = replace(content, "COLAB" => colab_header)
    return content
end
Sys.rm(notebooks_dir;recursive=true,force=true)
for (file,name) in files
  file_jl = file*".jl"
  Literate.notebook(joinpath(repo_src,file_jl), notebooks_dir; name=name, documenter=false, execute=false, preprocess=colab_install)
end

