module GeneralUtils
    # Libraries required
    using JLD2, FileIO
    using HDF5, JLD
    using NPZ
    using PyCall
    using PyPlot
    using Dates
    using Crayons

    @pyimport pickle

    #Module exports
    export check_rel_tol
    export save_file
    export load_file
    export write_to_file
    export show_all
    export show_matrix
    export delta_fn
    export fermi_fn
    export fermi_fn_classic
    export get_present_time
    export create_folders_if_not_exist
    """
        check_rel_tol(oldList, newList; tol=1e-5, nRound=10)
    Returns true/false if the two lists are within tolerance value
    """
    function check_rel_tol(oldList, newList; tol = 1e-5, nRound = 10)
        tolList = tol * ones(Float64, length(newList))
        relVals = (newList - oldList) ./ newList
        relTolList = [round(abs(i), digits = nRound) for i in relVals]
        flag = relTolList <= tolList
        return flag
    end # check_rel_tol end

    """
        save_file(object, fileName; key="data")
    Saves the given object in desired format(jld, jld2, pkl, npz)
    """
    function save_file(object, fileName; key = "data")
        if last(fileName, 3) == "jld"
            save(fileName, key, object)
        elseif last(fileName, 3) == "npz"
            npzwrite(fileName, object)
        elseif last(fileName, 4) == "jld2"
            save(fileName, key, object)
        elseif last(fileName, 3) == "pkl"
            f = open(fileName, "w")
            pickle.dump(object, f, protocol = pickle.HIGHEST_PROTOCOL)
            close(f)
        else
            throw("ERROR : Possibly wrong format ~ Try jld, jld2, npz or pkl")
        end

    end # save_file end

    """
        load_file(fileName; key="data")
    Loads the file from formats(npz, jld, jld2, pkl)
    """
    function load_file(fileName; key = "data")
        if last(fileName, 3) == "npz"
            mat = npzread(fileName)
        elseif last(fileName, 3) == "jld" || last(fileName, 4) == "jld2"
            mat = load(fileName)[key]
        elseif last(fileName, 3) == "pkl"
            # load the pickle file.
            f = open(fileName, "r")
            mat = pickle.load(f)
        else
            println("ERROR : $fileName not found")
        end
    end # load_file end



    """
        show_all(obj)
    shows the obj without any truncation
    """
    function show_all(obj)
        return show(stdout, "text/plain", obj)
    end

    function show_matrix(mat)
        PyPlot.gray()
        imshow(mat,interpolation="none")
        colorbar()
    end

    """
        delta_fn(i, j)
    Returns 0 if i ≠ j and 1 if i == j
    """
    function delta_fn(i, j)
        return i == j ?  1 : 0
    end

    """
        fermi_fn(E, mu=0; kwargs...)
    Returns the value of Fermi function for given value.
    kwargs : beta | β | T
    end
    """
    function fermi_fn(E, mu=0;kwargs...)
        if :beta in keys(kwargs)
            beta = kwargs[:beta]
            return (1 - tanh(0.5 * beta*(E-mu))) * 0.5
        elseif :β in keys(kwargs)
            β = kwargs[:β]
            return (1 - tanh(0.5 * β *(E-mu))) * 0.5
        elseif :T in keys(kwargs)
            T = kwargs[:T]
            return (1 - tanh(0.5 * (1/T)*(E-mu))) * 0.5
        else
            throw("Error: Provide beta, β or T")
        end
    end

    """
        fermi_fn_classic(E, T)
    Returns the value of Fermi function using the classic formula.
    """
    function fermi_fn_classic(E, T)
        return 1 / (exp(E / T) + 1)
    end

    """
        get_present_time()
    Returns present time.
    """
    function get_present_time()
        timeNow = Dates.format(now(), "HH:MM")
        dateToday = Dates.format(now(), DateFormat("d-m"))
        return timeNow, dateToday
    end

    """
        write_to_file(message, fileName; mode="a")
    Writes the message into the file and save it locally.
    """
    function write_to_file(message, fileName; mode="a")
        f = open(fileName, mode)
        write(f, message)
        close(f)
    end

    """
        create_folders_if_not_exist(folder_names::Vector{String}, base_path::AbstractString="")
    Creates folders if they do not exist
    """
    function create_folders_if_not_exist(folder_names::Vector{String}, base_path::AbstractString="")
        for folder_name in folder_names
            folder_path = joinpath(base_path, folder_name)
            if !isdir(folder_path)
                mkdir(folder_path)
                println(Crayon(foreground = :green, bold = true)("Folder created: $folder_path"))
            else
                println(Crayon(foreground = :yellow, bold = true)("Folder already exists: $folder_path"))
            end
        end
    end

end #module end
