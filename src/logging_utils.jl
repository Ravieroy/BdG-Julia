module LoggingUtils
    # Libraries required
    using Dates
    # Module exports
    export header_log
    export write_log
    # Includes
    include("../scripts/params.jl")
    include("../src/generalutils.jl")
    using .GeneralUtils

    function header_log(logFileName; kwargs...)
        seed = kwargs[:seed]
        np = kwargs[:np]
        nw = kwargs[:nw]
        pf = kwargs[:pf]
        dateFormat = "yyyy-mm-dd HH:MM:SS"
        startDate = Dates.format(now(), dateFormat)
        logSource = "This log file is generated from $pf\n"
        calcSummary = """
        Summary:
            α List = $alphaList
            Boundary Condition = $bc
            Correlated Disorder = $correlated
            Continued Calculation = $(continuedCalc)
            Dataset Folder = $dataSetFolder
            Lattice = $lattice
            Max Iteration = $maxCount
            N = $N
            Seed List = $seedList
            t = $t
            Temperature = $tempList
            Uncorrelated Disorder = $uncorrelated
            V Values = $vVals
        """
        message = """
            Started on : $startDate
            seed = $seed
            Number of processes = $(np)
            Number of Workers = $(nw)
        """

        headerString = string(logSource, calcSummary, message, "\n")
        write_to_file(headerString, logFileName)
    end


    function write_log(logFileName; kwargs...)
        timeNow, dateToday = get_present_time()
        defaults = Dict(:alpha => nothing,
                        :V => nothing,
                        :T => nothing,
                        :nJob => nothing,
                        :nt => nothing,
                        :endTime => nothing,
                        :count => nothing,
                        :nExp => nothing)

        for (key, default_value) in pairs(defaults)
            if !haskey(kwargs, key)
                kwargs = merge(kwargs, Dict(key => default_value))
            end
        end
        divider = "---------------------------------------------------\n"

        try
            if kwargs[:status] == "Running"
                α = kwargs[:alpha]
                V = kwargs[:V]
                T = kwargs[:T]
                nJob = kwargs[:nJob]
                nt = kwargs[:nt]
                nExp = kwargs[:nExp]
                logRunning = "($dateToday|$timeNow)" *
                             "$(kwargs[:status])" *
                             "($nJob/$nt) α = $α, V = $V, T = $T <n> = $nExp\n"
                write_to_file(logRunning, logFileName)
            elseif kwargs[:status] == "Converged"
                endTime = kwargs[:endTime]
                timeTaken = (round(endTime/60, digits = 2))
                count = kwargs[:count]
                logConverged = "($dateToday|$timeNow)" *
                               "$(kwargs[:status]) in " *
                               "$count iterations in $(timeTaken) mins\n"
                write_to_file(logConverged, logFileName)
                write_to_file(divider, logFileName)

            elseif kwargs[:status] == "HitMaxCount"
                endTime = kwargs[:endTime]
                timeTaken = (round(endTime/60, digits = 2))
                count = kwargs[:count]
                logHitMaxCount = "($dateToday|$timeNow)" *
                                 "$(kwargs[:status]) in " *
                                 "$count iteratons in $(timeTaken) mins\n"
                write_to_file(logHitMaxCount, logFileName)
                write_to_file(divider, logFileName)
            elseif kwargs[:status] == "Completed"
                endTime = kwargs[:endTime]
                timeTaken = (round(endTime/60, digits = 2))
                logCompleted = "($dateToday|$timeNow)" *
                               "$(kwargs[:status]) in $(timeTaken) mins\n"
                write_to_file(logCompleted, logFileName)
                write_to_file(divider, logFileName)
            end # if block end
        catch

        logUnknown = "($dateToday|$timeNow)Unknown: Possible error"
            write_to_file(logUnknown, logFileName)
        end # try-catch end
    end # function end
end # Module end
