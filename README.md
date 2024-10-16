The basic structure of the directory for running any BdG calculation is : 
```
├── `data`
├── `logs`
├── `results`
├── `scripts`
└── `src`
```

`data`: This directory is used to store all the important input files needed to run the BdG calculation. For example `df_square12.csv`, `vUncorrelatedDict12.pkl` etc.

`logs`: This directory is used to store log files.

`results`: This directory is used to store results from the BdG calculation.

`scripts`:  This is the directory where all the calculation is run. It contains all the script files needed to run any BdG calculation. 

```
scripts
├── `bdg_plotting_utils.py` : Plotting utilities(python) after completion BdG calculation.
├── `check_julia_dependency.jl` : Check and install Julia dependencies needed to run the calculation(`julia check_julia_dependency.jl`).
├── `check_python_dependency.py` : Check and python dependencies needed to run the calculation(`python check_python_dependency.jl`).
├── `create_input_files.jl` : This will create all the required input files needed to run the calculation.(`julia create_input_files.jl`).
├── `main_corr.jl` : This the main script to run BdG calculation for random correlated disorder.
├── `main_make_plots_T.py`: python script to plot important quantities after BdG calculation.
├── `main_uncorr.jl` : This the main script to run BdG calculation for random uncorrelated disorder.
├── `Makefile`: Makefile for running quick bash commands(`make reset` will clear everything except contents in the data directory. This is used to clear everything from previous calculation and run the fresh calculation with same input files. `make fresh` will delete all unnecessary files and run a fresh calculation. )
├── `params.jl`: Parameter file
├── `plot_params.py` : Parameter file for plotting results.
├── `scrift_info.md` : Markdown file containing summary of what each script does.
└── `show_status.sh` : Bash script to check the status of the running calculation.(`./show_status.sh` to check the status of all calculation(completed or running). `./show_status.sh -r` to check only calculation that is running.)
```

`src`: This directory contains all the source files needed to run a BdG calculation.

```
├── `bdg_utilities.jl` : All the utilities required for running BdG calculation
├── `external_utils.jl` : Some functions used for specific case, when Hamiltonian is taken from external source(e.g. FORTRAN). Usually not required for general calculation.
├── `generalutils.jl`: Contains all the general utilities which is useful for BdG calculation or other places.
└── `logging_utils.jl`: Utilities required for logging.
```
## Workflow 
1. **Step 0 :** Run Makefile for fresh calculation. `make fresh`
2. **Step 1 :** Edit the `params.jl` file accordingly 
3. **Step 2 :** Run `julia create_input_files.jl` to create input files needed to run the BdG calculation. It will create necessary directories and files.
4. **Step 3 :** Run `julia -p <n> script_name.jl` on `n` cores using the script `script_name.jl`


### Some results obtained using the code from the seminal paper by A.Ghosal et al. (PHYSICAL REVIEW B, VOLUME 65, 014501)

<table>
  <tr>
    <td><img src="assets/pairing_amplitude.png" alt="Pairing Amplitude Distribution" width="400"></td>
    <td><img src="assets/avg_n.png" alt="Local Electron Density Distribution" width="400"></td>
  </tr>
  <tr>
    <td><img src="assets/egap_deltagap.png" alt="Quasiparticle and Order Parameter Gap" width="400"></td>
    <td><img src="assets/DOS_V.png" alt="Density of States" width="400"></td>
  </tr>
</table>