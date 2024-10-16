include("../src/bdg_utilities.jl")
using .BdgUtilities

include("../src/generalutils.jl")
using .GeneralUtils

include("../scripts/params.jl")

# Create required folders
folder_names = ["data", "logs", "results", "scripts", "src", "post-processing"]
base_path = joinpath(pwd(), "..")
create_folders_if_not_exist(folder_names, base_path)

# Create dataframes only if nnHam is not prebuilt
if nnHam.type == "!prebuilt"
    saveDf = true
    dfName = string(dataSetFolder, "df_$lattice$N", ".csv")
    unitcell, lat, nnMap, df = create_nnmat_df(N, lattice, bc,
        saveAsCsv=saveDf,
        fileName=dfName)
else
    dfName = "NA(using prebuilt hamiltonnian)"
end

if correlated == true
    vDictName = string(dataSetFolder, "vDict_$lattice$N", ".", fileFormat)
    vDict = make_random_V_dict(lattice, seedList, alphaList, N, df)
    vCorrDict = get_corr_V_dict(lattice, seedList, alphaList, vVals, N, df)
    save_file(vCorrDict, vDictName; key="data")
    fileName = vDictName
elseif uncorrelated == true
    vUncorrelatedDict = create_uncorrelated_disorder(
        nSites,
        seedList,
        vVals,
        pyRandom=true)
    vUncorrelatedDictName = string(
        dataSetFolder,
        "vUncorrelatedDict$(N)", ".", fileFormat)
    save_file(vUncorrelatedDict, vUncorrelatedDictName)
    fileName = vUncorrelatedDictName
else
    throw("""
        correlated, partial, sublattice are false.
        For uncorrelated disorder on all the sites, use
        partial = true, partialDisorderOn = [true, true, true]
        or uncorrelated = true
        """)
end

println("""
    Summary:
    lattice = $lattice
    N = $N
    bc = $bc
    correlated = $correlated
    uncorrelated = $uncorrelated
    dataframe = $dfName
    dataset Folder = $dataSetFolder
    filename = $fileName
""")
