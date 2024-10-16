module BdgUtilities
using LatticeUtilities
using DelimitedFiles
using PyCall
using LinearAlgebra
using CSV
using Distributions
using DataFrames
using Random
@pyimport numpy as np
@pyimport random
include("../src/generalutils.jl")
using .GeneralUtils

#Module exports
export create_lattice_df
export create_nnmat_df
export create_t_matrix
export create_bdg_ham
export sort_evecs
export check_norm
export calc_delta
export calc_avg_n
export check_rel_tol
export run_self_consistency_numpy
export get_power_law_i_kagome
export get_power_law_i_square
export get_random_V
export make_random_V_dict
export get_scaled_V
export get_corr_V_dict
export get_effective_V
export create_uncorrelated_disorder
export create_complex_ham

# alias to Fermi function
const f = fermi_fn

"""
    create_lattice_df(nnMap, unitCell, lattice, nSites; saveInFile=nothing, fileName=nothing)
Returns a dataframe containing all the information related to lattice.
"""
function create_lattice_df(nnMap, unitCell, lattice, nSites; saveInFile=nothing, fileName=nothing)
    df = DataFrame(siteIndex=Int[],
        sitePosX=Float64[],
        sitePosY=Float64[],
        first=Int[],
        second=Int[],
        third=Int[],
        fourth=Int[]
    )
    for site in 1:nSites
        arr = zeros(ncol(df))
        loc = site_to_loc(site, unitCell, lattice)
        n1 = loc[1][1]
        n2 = loc[1][2]
        subLat = loc[2]
        pos = loc_to_pos([n1, n2], subLat, unitCell)
        arr[1] = site
        arr[2] = round(pos[1], digits=3)
        arr[3] = round(pos[2], digits=3)
        for i in 1:length(nnMap[1][2])
            arr[i+3] = nnMap[site][2][i]
        end
        push!(df, arr)
    end
    if saveInFile == true
        if fileName !== nothing
            CSV.write(fileName, df)
        else
            fileName = "../data/NN_MAP.csv"
            println("saving as $fileName")
            CSV.write(fileName, df)
        end
    end
    return df
end

"""
    create_nnmat_df(N, lattice, bc; saveAsCsv=nothing, fileName=nothing)
Returns the nearest neighbor map and dataframe of the given lattice.
The dataframe can be saved as csv file.
"""
function create_nnmat_df(N, lattice, bc; saveAsCsv=nothing, fileName=nothing)
    if lattice == "square"
        nSites = N^2
        latticeVecs = [[1, 0.0], [0.0, 1]]
        basisVecs = [[0.0, 0.0]]
        square = UnitCell(latticeVecs, basisVecs)
        if bc == "pbc"
            periodic = [true, true]
        elseif bc == "obc"
            periodic = [false, false]
        else
            throw("Wrong boundary condition.(pbc || obc)")
        end
        L = [N, N]
        lat = Lattice(L, periodic)
        bondX = Bond(orbitals=(1, 1), displacement=[1, 0])
        bondY = Bond((1, 1), [0, 1])
        nbrTable = build_neighbor_table([bondX, bondY], square, lat)
        nbrTableMap = map_neighbor_table(nbrTable)
        df = create_lattice_df(nbrTableMap,
            square,
            lat,
            nSites;
            saveInFile=saveAsCsv,
            fileName=fileName)
        return square, lat, nbrTableMap, df

    elseif lattice == "kagome"
        nSites = 3 * N^2
        latticeVecs = [[1.0, 0.0], [1 / 2, √3 / 2]]
        basisVecs = [[0.0, 0.0], [1 / 2, 0.0], [1 / 4, √3 / 4]]
        kagome = UnitCell(latticeVecs, basisVecs)
        if bc == "pbc"
            periodic = [true, true]
        elseif bc == "obc"
            periodic = [false, false]
        else
            throw("Wrong boundary condition.(pbc || obc)")
        end
        L = [N, N]
        lat = Lattice(L, periodic)
        bond_1 = Bond(orbitals=(1, 2), displacement=[0, 0])
        bond_2 = Bond((1, 3), [0, 0])
        bond_3 = Bond((2, 3), [0, 0])
        bond_4 = Bond((2, 1), [1, 0])
        bond_5 = Bond((3, 1), [0, 1])
        bond_6 = Bond((3, 2), [-1, 1])
        nbrTable = build_neighbor_table(
            [bond_1, bond_2, bond_3, bond_4, bond_5, bond_6],
            kagome,
            lat)
        nbrTableMap = map_neighbor_table(nbrTable)
        df = create_lattice_df(nbrTableMap,
            kagome,
            lat,
            nSites;
            saveInFile=saveAsCsv,
            fileName=fileName)
        return kagome, lat, nbrTableMap, df
    else
        throw("No implementation for $lattice(square || kagome)")
    end
end


"""
        create_t_matrix(lattice, N, nnMap; t=1, saveAsText=nothing, fileName=nothing)
    Return hopping matrix for the given `lattice` and its parameters.
    The t-matrix can be saved as text locally by setting `saveAsText=true`
    with `fileName=tMatFileName` in the params file.

    """
function create_t_matrix(nnMap; dataSetFolder=nothing, t=1, saveAsText=nothing, fileName=nothing)
    nSites = length(nnMap)
    H = zeros(Float64, nSites, nSites)
    for site in 1:nSites
        for nbr in nnMap[site][2]
            H[site, nbr] = -t
        end
    end

    if saveAsText == true
        if fileName !== nothing && dataSetFolder !== nothing
            fullFileName = joinpath(dataSetFolder, fileName)
            CSV.write(fullFileName, DataFrame(H, :auto))

        else
            fullFileName = joinpath(dataSetFolder, fileName)
            println("saving as $fullFileName")
            CSV.write(fullFileName, DataFrame(H, :auto))
        end
    end
    return H
end


"""
    create_bdg_ham(deltaList, H, mu, nAvgOld, nSites, U, vList)
Returns the BdG Hamiltonian for given lattice.
"""
function create_bdg_ham(deltaList, H, mu, nAvgOld, nSites, U, vList)
    for i in 1:nSites
        H[i, i] = -mu - (U / 2) * nAvgOld[i] + vList[i]
    end

    # Making BdG Hamiltonian
    HBdG = zeros(Float64, 2 * nSites, 2 * nSites)
    for i in 1:nSites
        for j in 1:nSites
            HBdG[i, j] = H[i, j]
        end
    end

    for i in nSites+1:2*nSites
        for j in nSites+1:2*nSites
            HBdG[i, j] = -conj(H[i-nSites, j-nSites])
        end
    end

    # Add delta term
    for i in 1:nSites
        HBdG[i, i+nSites] = deltaList[i]
    end

    for i in nSites+1:2*nSites
        HBdG[i, i-nSites] = deltaList[i-nSites]
    end
    return HBdG
end


"""
    sort_evecs(evector, nSites)
Julia gives the eigenvectors in columns.
i.e. First column is eigenvector for first eigenvalue and so on.
evecs[:, n] gives the eigenvectors for nth eigenvalue.
For BdG we want En>0 which starts from nSites+1 to 2*nSites.

What we want is that all the rows contain eigenvectors.
"""
function sort_evecs(evector, nSites)
    # Transpose to rearrange the vectors for easier slicing
    evecs = transpose(evector)[nSites+1:2*nSites, :]

    # Extract un and vn directly from the matrix
    un = evecs[:, 1:nSites]
    vn = evecs[:, nSites+1:2*nSites]

    return un, vn
end


"""
    check_norm(un, vn, nSites)
Checks norm |un^2| + |vn^2| = 1 at every site for each eigenvalue, n
"""
function check_norm(un, vn, nSites)
    normList = zeros(Float64, nSites)
    for n in 1:size(un)[1] # gives the number of rows in un, i.e. n
        val = dot(un[n, :], un[n, :]) + dot(vn[n, :], vn[n, :])
        normList[n] = val
    end
    return round.(normList, digits=5)
end

"""
    calc_delta(U, un, vn, nSites)
Calculates delta for the given un and vn
"""
function calc_delta(U, un, vn, nSites, evals, T)
    deltaList = zeros(Float64, nSites)
    for i in 1:nSites
        delta = 0
        for n in 1:size(un)[1] # gives the number of rows in un, i.e. nth eigenvalue
            E = evals[n]
            # fn = f(E, T)
            fn = f(E, T=T)
            delta += U * (un[n, i] * conj(vn[n, i])) * (1 - 2 * fn)
            # delta += U * (un[n, i] * conj(vn[n, i]))
        end
        deltaList[i] = delta
    end
    return deltaList
end

"""
    calc_delta(U, un, vn, nSites)
Calculates N-average for the given vn
"""
function calc_avg_n(un, vn, nSites, evals, T)
    nAvgList = zeros(Float64, nSites)
    for i in 1:nSites
        nAvg = 0
        for n in 1:size(vn)[1] # gives the number of rows in un, i.e. nth eigenvalue
            E = evals[n]
            # fn = f(E, T)
            fn = f(E, T=T)
            nAvg += ((un[n, i] * conj(un[n, i])) * fn + ((vn[n, i] * conj(vn[n, i])) * (1 - 2 * fn)))
        end
        nAvgList[i] = nAvg
    end
    return 2 * nAvgList
end

"""
    run_self_consistency_numpy(T, deltaOld, mu, nSites, nAvgOld, nExp, tMat, U, vList; tol=0.001, maxCount=300)
    Runs the self-consistent loop for BdG Hamiltonian using numpy module of Python.
"""
function run_self_consistency_numpy(T, deltaOld, mu, nSites, nAvgOld, nExp, tMat, U, vList; tol=0.001, maxCount=300)
    startTime = time()
    count = 0
    flag = false
    while flag == false
        count += 1
        H = create_bdg_ham(deltaOld, tMat, mu, nAvgOld, nSites, U, vList)
        (evals, evecs) = np.linalg.eigh(H)
        un, vn = sort_evecs(evecs, nSites)
        evalPostive = evals[nSites+1:end]
        deltaNew = calc_delta(U, un, vn, nSites, evalPostive, T)
        nAvgNew = calc_avg_n(un, vn, nSites, evalPostive, T)
        nAvg = sum(nAvgNew) / nSites
        mu += 0.3 * (nExp - nAvg)
        deltaFlag = isapprox(deltaOld, deltaNew, rtol=tol)
        nFlag = isapprox(nAvgOld, nAvgNew, rtol=tol)
        nAvgFlag = isapprox(nExp, nAvg, rtol=1e-4)
        if deltaFlag == true && nFlag == true && nAvgFlag == true
            deltaFinal = deltaNew
            nAvgFinal = nAvgNew
            eGap = minimum(evals[nSites+1:2*nSites])
            deltaGap = sum(deltaFinal) / nSites
            endTime = round(time() - startTime, digits=2)
            isConverged = true
            return deltaFinal, nAvgFinal, eGap, deltaGap, evecs, evals, count, endTime, isConverged
            break
        else
            deltaOld = deltaNew
            nAvgOld = nAvgNew
            if count >= maxCount
                deltaFinal = deltaNew
                nAvgFinal = nAvgNew
                eGap = minimum(evals[nSites+1:2*nSites])
                deltaGap = sum(deltaFinal) / nSites
                endTime = round(time() - startTime, digits=2)
                isConverged = false
                flag = true
                return deltaFinal, nAvgFinal, eGap, deltaGap, evecs, evals, count, endTime, isConverged
                break
            end
        end
    end #while loop end
end # function end


"""
    run_self_consistency(T, deltaOld, mu, nSites, nAvgOld, nExp, tMat, U, vList; tol=0.001, maxCount=300)
Runs the self-consistent loop for BdG Hamiltonian.
"""
function run_self_consistency(T, deltaOld, mu, nSites, nAvgOld, nExp, tMat, U, vList; tol=0.001, maxCount=300)
    startTime = time()
    count = 0
    flag = false
    while flag == false
        count += 1
        H = create_bdg_ham(deltaOld, nSites, tMat, mu, nAvgOld, U, vList)
        (evals, evecs) = eigen(H)
        un, vn = sort_evecs(evecs, nSites)
        evalPostive = evals[nSites+1:end]
        deltaNew = calc_delta(U, un, vn, nSites, evalPostive, T)
        nAvgNew = calc_avg_n(un, vn, nSites, evalPostive, T)
        nAvg = sum(nAvgNew) / nSites
        mu += 0.3 * (nExp - nAvg)
        deltaFlag = isapprox(deltaOld, deltaNew, rtol=tol)
        nFlag = isapprox(nAvgOld, nAvgNew, rtol=tol)
        nAvgFlag = isapprox(nExp, nAvg, rtol=1e-4)
        if deltaFlag == true && nFlag == true && nAvgFlag == true
            deltaFinal = deltaNew
            nAvgFinal = nAvgNew
            eGap = minimum(evals[nSites+1:2*nSites])
            deltaGap = sum(deltaFinal) / nSites
            endTime = round(time() - startTime, digits=2)
            isConverged = true
            return deltaFinal, nAvgFinal, eGap, deltaGap, evecs, evals, count, endTime, isConverged
            break
        else
            deltaOld = deltaNew
            nAvgOld = nAvgNew
            if count >= maxCount
                deltaFinal = deltaNew
                nAvgFinal = nAvgNew
                eGap = minimum(evals[nSites+1:2*nSites])
                deltaGap = sum(deltaFinal) / nSites
                endTime = round(time() - startTime, digits=2)
                isConverged = false
                flag = true
                return deltaFinal, nAvgFinal, eGap, deltaGap, evecs, evals, count, endTime, isConverged
                break
            end
        end
    end #while loop end
end # function end


"""
   get_power_law_i_kagome(alpha, df, N, seed, site)
Returns random potential for a site from the prescription
in Communications Physics, 5, 177 (2022)
"""
function get_power_law_i_kagome(alpha, df, N, seed, site)
    ri = [df[site, :].sitePosX, df[site, :].sitePosY]
    res = 0
    np.random.seed(seed)
    for jx in 1 : N / 2
        for jy in 1 : N / 2
            phij = np.random.uniform(0, 2 * pi)
            qj = ((2 * pi * jx) / N, (2 * pi * jy) / N)
            factor = norm(qj, 2)^(-alpha / 2)
            res += factor * cos(dot(qj, ri) + phij)
        end
    end
    return res / (3 * N ^ 2)
end

"""
   get_power_law_i_square(alpha, df, N, seed, site)
Returns random potential for a site from the prescription
in Communications Physics, 5, 177 (2022)
"""
function get_power_law_i_square(alpha, df, N, seed, site)
    ri = [df[site, :].sitePosX, df[site, :].sitePosY]
    res = 0
    np.random.seed(seed)
    for jx in 1 : N / 2
        for jy in 1 : N / 2
            phij = np.random.uniform(0, 2 * pi)
            qj = ((2 * pi * jx) / N, (2 * pi * jy) / N)
            factor = norm(qj, 2)^(-alpha / 2)
            res += factor * cos(dot(qj, ri) + phij)
        end
    end
    return res / (N ^ 2)
end

"""
   get_random_V(alpha, df, N, seed)
Returns an array with correlated random numbers for a given
lattice(df),alpha, N and seed
"""
function get_random_V(lattice, alpha, df, N, seed)
    nSites = nrow(df)
    vRandomList = zeros(Float64, nSites)
    if lattice == "square"
        for site in 1 : nSites
            res = get_power_law_i_square(alpha, df, N, seed, site)
            vRandomList[site] = res
        end
    elseif lattice == "kagome"
        for site in 1 : nSites
            res = get_power_law_i_kagome(alpha, df, N, seed, site)
            vRandomList[site] = res
        end
    end
    return vRandomList
end

"""
    make_random_V_dict(seedList, alphaList, N, df)
Returns a dictionary containing correlated random disorders
(not normalized) for an array of seed and alpha.
dict[seed][alpha]
"""
function make_random_V_dict(lattice, seedList, alphaList, N, df)
    vDict = Dict()
    for seed in seedList
        vDictAlpha = Dict()
        for alpha in alphaList
            vList = get_random_V(lattice, alpha, df, N, seed)
            vDictAlpha[alpha] = vList
        end # alpha loop end
        vDict[seed] = vDictAlpha
    end # seed loop end
    return vDict
end #function end

"""
    get_scaled_V(vList, V)
Returns the scaled list of correlated random disorders for the given V
"""
function get_scaled_V(vList, V)
    rMin = minimum(vList)
    rMax = maximum(vList)

    tMin = -V
    tMax = V

    vNorm = (vList .- rMin) ./ (rMax - rMin) .* (tMax - tMin) .+ tMin
    return vNorm
end

"""
   get_corr_V_dict(seedList, alphaList, vList, N, df)
Returns a dictionary containing correlated random disorders
(normalized) for an array of seed, alpha and V.
dict[seed][alpha][V]
"""
function get_corr_V_dict(lattice, seedList, alphaList, vList, N, df)
    vDict = make_random_V_dict(lattice, seedList, alphaList, N, df)
    vCorrSeed = Dict()
    count = 0
    effVDict = Dict()
    for seed in seedList
        vCorrAlpha = Dict()
        for alpha in alphaList
            Vi = vDict[seed][alpha]
            vCorrV = Dict()
            for V in vList
                count += 1
                effV = get_effective_V(Vi, V)
                #---- Storing ----------
                effArr = [seed, V, alpha, effV]
                effVDict[count] = effArr
                #------end storing------
                vNorm = get_scaled_V(Vi, effV)
                vCorr = vNorm .- sum(vNorm) / length(vNorm)
                vCorrV[V] = vCorr
            end # V loop end
            vCorrAlpha[alpha] = vCorrV
        end # alpha loop end
        vCorrSeed[seed] = vCorrAlpha
    end # seed loop end
    effVname = "../data/effV"
    writedlm(effVname, sort(effVDict))
    open(effVname, "a") do file
        write(file, "count [seed, V, alpha, effV]")
    end
    return vCorrSeed
end # function end

"""
    get_effective_V(vList, V)
Returns the effective value for disorder for the given V.
This function normalises and then fits the data with Normal distribution
and returns the standard deviation.
"""
function get_effective_V(vList, V)
    for val in 0:0.0001:15
        vNorm = get_scaled_V(vList, val)
        vCorr = vNorm .- sum(vNorm) / length(vNorm)
        fitParams = fit(Normal, vCorr)
        sigma = fitParams.σ
        if isapprox(sigma, V; rtol=0.001)
            return val
            break
        end
    end
end


"""
     create_uncorrelated_disorder(N, V, seedList, vVals; pyRandom=true)
Returns a dictionary with uncorrelated disorders for the given seed and V values
"""
function create_uncorrelated_disorder(nSites, seedList, vVals; pyRandom=true)
    vUncorrelatedDict = Dict()
    vList = zeros(nSites)
    for seed in seedList
        vUncorrelatedDictV = Dict()
        for V in vVals
            if pyRandom == true
                np.random.seed(seed)
                vList = np.random.uniform(-V, V, nSites)
                vUncorrelatedDictV[V] = vList
            else
                Random.seed!(seed)
                vList = rand(Uniform(-V, V), nSites)
                vUncorrelatedDictV[V] = vList
            end # pyRandom if block end
        end # V for loop end
        vUncorrelatedDict[seed] = vUncorrelatedDictV
    end # seed for loop end
    return vUncorrelatedDict
end


"""
    create_complex_ham(fileName)
Takes Hamiltonian made from FORTRAN code and returns the complex Hamiltonian
"""
function create_complex_ham(fileName)
    rawHam = Matrix(readdlm(fileName))
    nrows = size(rawHam)[1]
    ncols = size(rawHam)[1]
    H = zeros(Complex{Float64}, nrows, ncols)
    for row = 1:nrows
        c = 1 # counter
        for col = 1:ncols
            H[row, col] = rawHam[row, c] + im * rawHam[row, c+1]
            c += 2 # adding 2 so that we have pairs like (1, 2), (3, 4) etc.
        end # col loop end
    end # row loop end
    return H
end # function end

end #module end
