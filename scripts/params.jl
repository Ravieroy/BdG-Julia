N                       =  8
nnb                     =  4
# lattice                 =  "kagome"
lattice                 =  "square"
# nSites                  = 3 * N ^ 2
nSites                  =  N ^ 2
a                       =  1
t                       =  1
bc                      =  "pbc"
tMatSave                =  false
tMatFileName            =  "tmat_$lattice$N.csv"
nExp                    =  0.875
tol                     =  0.001
nAvgOld                 =  ones(Float64, nSites)
deltaOld                =  ones(Float64, nSites)
maxCount                =  200
mu                      =  -0.92
U                       =  1.5  * t
tempList                =  [0]
seedList                =  1:2:9
vVals                   =  [0.1, 3]
alphaList               =  [0]
correlated              =  false
uncorrelated            =  true
continuedCalc           = (state = false, variable = "V", run=2)
nnHam                   =  (type =  "!prebuilt", fileName =  "../data/ham_$N")
flag                    =  false
dataSetFolder           =  "../data/"
logFolder               =  "../logs/"
fileFormat              =  "pkl"
saveInFolder            =  "../results/"
store                   =  Dict([
    ("store_delta_op", [true, "deltaDict", 1]),
    ("store_avg_n", [true, "nAvgDict", 2]),
    ("store_egap", [true, "eGapDict", 3]),
    ("store_delta_gap", [true, "deltaGapDict", 4]),
    ("store_evectors", [true, "evectors", 5]),
    ("store_evalues", [true, "evalues", 6]),
])
