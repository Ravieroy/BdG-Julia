using Pkg

library_list = [
    "CSV",
    "DataFrames",
    "DelimitedFiles",
    "Distributions",
    "FileIO",
    "HDF5",
    "JLD",
    "JLD2",
    "KernelDensity",
    "LatticeUtilities",
    "LinearAlgebra",
    "NPZ",
    "PyCall",
    "PyCall",
    "PyPlot",
    "Random",
    "StatsPlots",
]

io = IOBuffer()
Pkg.status(; io=io)
st = String(take!(io))

postiveResponses = ["Y", "y", "yes", "Yes"]
negativeResponses = ["N", "n", "no", "No"]

function install_package(list; installAll=true)
    if installAll == true
        println("Installing all missing packages\n $list")
        Pkg.add(list)
    else
        for package in list
            println("Install $package(y|n): ")
            response = readline()
            if response ∈ postiveResponses
                println("Installing $package")
                Pkg.add(package)
            elseif response ∈ negativeResponses
                println("Skipping Install $package")
            else
                println("Wrong Input: Try Again")
            end
        end
    end
end

function run_installation_wizard(response)
    if response ∈ postiveResponses
        @show(missing_packages)
        println("Do you want to install all missing packages? (y|n) : ")
        response = readline()
        if response ∈ postiveResponses
            install_package(missing_packages, installAll=true)
        elseif response ∈ negativeResponses
            install_package(missing_packages, installAll=false)
        end
    elseif response ∈ negativeResponses
        println("Skipping Installation")
    end
end



missing_packages = String[]

for name in library_list
    if contains(st, name) == false
        push!(missing_packages, name)
    end
end

if missing_packages != []
    @show(missing_packages)
    println("Do you want to start installation wizard? (y/n) : ")
    response = readline()
    run_installation_wizard(response)
else
    println("All dependencies are satisfied")
end



