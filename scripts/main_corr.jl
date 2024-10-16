# Importing required libraries
using Distributed
@everywhere begin
    using LatticeUtilities
    using DelimitedFiles
    using LinearAlgebra
    using Dates
    using PyCall
    @everywhere(@pyimport numpy as np)
    @everywhere(@pyimport random)
    # Importing personal modules
    include("../src/bdg_utilities.jl")
    include("../src/generalutils.jl")
    include("../src/logging_utils.jl")
    include("params.jl")
end # module everywhere end

# Loading personal modules
@everywhere using .BdgUtilities
@everywhere using .GeneralUtils
@everywhere using .LoggingUtils

@everywhere function main(seed)
    #------Logging part-----------
    logFileName = string(logFolder,
        "info",
        "_seed_$seed",
        "_id_$(myid())",
        ".log"
    )
    np = nprocs()
    nw = nworkers()
    pf = @__FILE__
    st = time()
    header_log(logFileName, seed=seed, np=np, nw=nw, pf=pf)
    #-----------Logging part end ------
    nTotalJobs = length(alphaList) * length(vVals) * length(tempList)
    nJob = 0
    deltaDict = Dict()
    nAvgDict = Dict()
    eGapDict = Dict()
    deltaGapDict = Dict()
    evecsDict = Dict()
    evalsDict = Dict()
    for alpha in alphaList
        deltaDictV = Dict()
        nAvgDictV = Dict()
        eGapDictV = Dict()
        deltaGapDictV = Dict()
        evecsDictV = Dict()
        evalsDictV = Dict()
        for V in vVals
            deltaDictT = Dict()
            nAvgDictT = Dict()
            eGapDictT = Dict()
            deltaGapDictT = Dict()
            evecsDictT = Dict()
            evalsDictT = Dict()
            for T in tempList
                nJob += 1
                #------Logs-------
                write_log(logFileName,
                    status="Running",
                    alpha=alpha,
                    V=V,
                    T=T,
                    nJob=nJob,
                    nt=nTotalJobs,
                    nExp=nExp
                )
                vList = vDict[seed][alpha][V]

                deltaFinal, nAvgFinal, eGap, deltaGap, evecs, evals, count, endTime, isConverged =
                    run_self_consistency_numpy(T,
                        deltaOld,
                        mu,
                        nSites,
                        nAvgOld,
                        nExp,
                        tMat,
                        U,
                        vList,
                        tol=tol,
                        maxCount=maxCount)
                if isConverged == true
                    write_log(logFileName,
                        status="Converged",
                        count=count,
                        endTime=endTime
                    )
                else
                    write_log(logFileName,
                        status="HitMaxCount",
                        count=count,
                        endTime=endTime
                    )
                end
                deltaDictT[T] = deltaFinal
                nAvgDictT[T] = nAvgFinal
                eGapDictT[T] = eGap
                deltaGapDictT[T] = deltaGap
                evecsDictT[T] = evecs
                evalsDictT[T] = evals
            end # T loop end
            deltaDictV[V] = deltaDictT
            nAvgDictV[V] = nAvgDictT
            eGapDictV[V] = eGapDictT
            deltaGapDictV[V] = deltaGapDictT
            evecsDictV[V] = evecsDictT
            evalsDictV[V] = evalsDictT
        end #V loop end
        deltaDict[alpha] = deltaDictV
        nAvgDict[alpha] = nAvgDictV
        eGapDict[alpha] = eGapDictV
        deltaGapDict[alpha] = deltaGapDictV
        evecsDict[alpha] = evecsDictV
        evalsDict[alpha] = evalsDictV
    end # alpha loop end
    # code block to save dictionary locally
    dictList = [deltaDict,
        nAvgDict,
        eGapDict,
        deltaGapDict,
        evecsDict,
        evalsDict
    ]
    for (key, value) in store
        if value[1] == true
            baseFileName = value[2]
            if continuedCalc.state == true
                run = continuedCalc.run
                var = continuedCalc.variable
                baseFileName = string(baseFileName, "Run", var, run)
            end # continuedCalc block end
            newFileName = string(saveInFolder,
                baseFileName,
                "_",
                seed,
                ".", fileFormat
            )
            save_file(dictList[value[3]], newFileName)

        end
    end # store loop end
    et = round(time() - st, digits=2)
    write_log(logFileName, status="Completed", endTime=et)
    # free up memory
    deltaDict = nothing
    nAvgDict = nothing
    eGapDict = nothing
    deltaGapDict = nothing
    evecsDict = nothing
    evalsDict = nothing

    deltaDictT = nothing
    nAvgDictT = nothing
    eGapDictT = nothing
    deltaGapDictT = nothing
    evecsDictT = nothing
    evalsDictT = nothing

    deltaDictV = nothing
    nAvgDictV = nothing
    eGapDictV = nothing
    deltaGapDictV = nothing
    evecsDictV = nothing
    evalsDictV = nothing
    GC.gc()
end  # main end

timestart = time()
if nnHam.type == "prebuilt"
    @everywhere begin
        vDictName = string(dataSetFolder,
            "vDict_$lattice$N",
            ".", fileFormat
        )
        vDict = load_file(vDictName)
        tMat = readdlm(nnHam.fileName)
    end
elseif nnHam.type == "!prebuilt"
    @everywhere begin
        #nnMap[site][2] = nearest neighbors
        _, _, nnMap, df = create_nnmat_df(N, lattice, bc)
        vDictName = string(dataSetFolder,
            "vDict_$lattice$N",
            ".", fileFormat
        )
        vDict = load_file(vDictName)
        tMat = create_t_matrix(nnMap,
            t = t,
            saveAsText = tMatSave,
            fileName = tMatFileName,
            dataSetFolder = dataSetFolder
        )
    end
else
    throw("ERROR : choose nnHam.type to be prebuilt or !prebuilt")
end # if block end

pmap(main, seedList)
elapsed = round(time() - timestart, digits=2)
timeNow = Dates.format(now(), "HH:MM")
roundedElapsed = round(elapsed / 60, digits=2)
println("($timeNow)The elapsed time : $elapsed secs ($roundedElapsed mins)")
