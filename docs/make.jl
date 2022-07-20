using SpeedyExperiments
using Documenter

DocMeta.setdocmeta!(SpeedyExperiments, :DocTestSetup, :(using SpeedyExperiments); recursive=true)

makedocs(;
    modules=[SpeedyExperiments],
    authors="Maximilian Gelbrecht <maximilian.gelbrecht@posteo.de> and contributors",
    repo="https://github.com/maximilian-gelbrecht/SpeedyExperiments.jl/blob/{commit}{path}#{line}",
    sitename="SpeedyExperiments.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://maximilian-gelbrecht.github.io/SpeedyExperiments.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/maximilian-gelbrecht/SpeedyExperiments.jl",
    devbranch="main",
)
