module ExternalUtils
    using DelimitedFiles
    using DataFrames
    using DataStructures # for SortedDict
    using CSV
    include("../scripts/params.jl")


    export init_from_file
    export create_nbr_dict
    export get_index
    export nbr_of
    export create_nnMat
    export create_df_ML
    export create_df_BL


    function init_from_file(fileName)
        nLayer = 1
        nL = nLayer
        NN_MAP = readdlm(fileName)
        # transpose because julia uses column major for matrix operations.
        COORD = transpose(transpose(NN_MAP)[1 : nL+1, :])
        nnMap = NN_MAP/1000
        coord = COORD/1000;
        return nnMap, coord
    end

    function create_nbr_dict(nnMap)
        nSites = size(nnMap)[1]
        nbrCoordDict = SortedDict()
        nL = nLayer
        # block to assert we have right nnb
        if nL == 1
            @assert(nnb == 4)
        elseif nL == 2
            @assert(nnb == 5)
        else
            throw("Works only for nL=1 or nL=2")
        end

        for i in 1:nSites
            tmpDict = SortedDict()
            for nbr in 1:nnb
                # for ex. for monolayer nL=1
                # nnMap[i,:](2*nbr+1 : 2*nbr+2)
                tmpDict[nbr] = nnMap[i , :][(nL+1) * nbr + 1 : (nL+1) * nbr + (nL+1)]
            end
            nbrCoordDict[i] = tmpDict
        end
        return nbrCoordDict
    end

    function get_index(subVec, vec)
        try
            for idx in 1:length(vec)
                if subVec == vec[idx, :]
                    return idx
                end
            end
        catch BoundsError
            return -1
        end
    end


    function nbr_of(site, nnMap)
        nL = nLayer
        coord = transpose(transpose(nnMap)[1 : nL+1, :])
        nbrCoordDict = create_nbr_dict(nnMap)
        nbrList = zeros(nnb)
        for nbr in 1:nnb
            val = nbrCoordDict[site][nbr]
            idx = get_index(val, coord)
            nbrList[nbr] = idx
        end
        return nbrList
    end

    function create_nnMat(nnMap)
        nL = nLayer
        nSites = nL * 3 * N ^ 2
        nnMat = SortedDict()
        for site in 1:nSites
            nnMat[site] = nbr_of(site, nnMap)
        end
        return nnMat
    end

    function create_df_ML(nnMapFileName; shouldSaveLocally=true, fileName=nothing)
        nnMap, coord = init_from_file(nnMapFileName)
        nnMat = create_nnMat(nnMap)
        df = DataFrame(siteIndex = Int[],
            sitePosX = Float64[],
            sitePosY = Float64[],
            first = Int[],
            second = Int[],
            third = Int[],
            fourth = Int[]
            )
        for site in 1:nSites
            arr = zeros(ncol(df))
            arr[1] = site
            arr[2] = coord[site, 1]
            arr[3] = coord[site, 2]
            for i in 1:nnb
                arr[i+3] = nnMat[site][i]
            end
            push!(df, arr)
        end

        if shouldSaveLocally == true
            if fileName !== nothing
                CSV.write(fileName, df)
            else
                fileName = string("data/","df_$lattice$N.csv")
                println("saving as $fileName")
                CSV.write(fileName, df)
            end
        end
        return df
    end # function end


function create_df_BL(nnMapFileName; shouldSaveLocally=true, fileName=nothing)
        nnMap, coord = init_from_file(nnMapFileName)
        nnMat = create_nnMat(nnMap)

        df = DataFrame(siteIndex = Int[],
            sitePosX = Float64[],
            sitePosY = Float64[],
            layer = Int[],
            first = Int[],
            second = Int[],
            third = Int[],
            fourth = Int[],
            fifth = Int[]
            )
        for site in 1:nSites
            arr = zeros(ncol(df))
            arr[1] = site
            arr[2] = coord[site, 1]
            arr[3] = coord[site, 2]
            arr[4] = 1000 * coord[site, 3]
            for i in 1:nnb
                arr[i+4] = nnMat[site][i]
            end
            push!(df, arr)
        end

        if shouldSaveLocally == true
            if fileName !== nothing
                CSV.write(fileName, df)
            else
                fileName = string("data/","df_$lattice$N.csv")
                println("saving as $fileName")
                CSV.write(fileName, df)
            end
        end
        return df
    end # function end
end # module end


