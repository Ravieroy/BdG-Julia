N = 24
# lattice = "kagome"
lattice = "square"
n_layer = 1
# n_sites = 3 * N ** 2
n_sites = N ** 2
a = 1
t = 1
n_exp = 0.875
tol = 0.001
mu = -0.9
U = 1.5 * t
seed_list = range(1, 11, 2)
v_vals_plot = [0.1, 1, 2, 3]
T = 0  # used for plotting P(Δ), P(n), DOS, Egap, P(Δ=0)
temp_list = [0]
alpha_list_plot = [0]
data_set_folder = "../data/"
file_format = "pkl"
results_folder = "../results/"
logFolder = "../logs/"
dict_list = ["deltaDict", "nAvgDict", "eGapDict", "deltaGapDict"]
post_processing_folder = "../post-processing/"
images_folder = "../images/"
created_collated_files = True
update_collated_files = False
df_filename = f"{data_set_folder}df_{lattice}{N}.csv"
interval = 5

info_continued_calc = {
    "state": False,
    "var": "V",
    "run": 2,
    "var_values": [0.5, 2],
    "dict_list": dict_list,
    "seed_list": seed_list,
    "alpha_list": alpha_list_plot,
    "v_vals": v_vals_plot,
}

info_delta_op_plot = {
    "skip_calculation": False,
    "file_name": "deltaDict.pkl",
    "vline": {"show": True, "val": 0.153},
    "x_lim": [0, 0.2],
    "text_size": 15,
    "T": T,
    "bw_list": [3, 3, 0.5, 0.5, 0.5, 0.5, 0.5],
    "save_as": "delta_op",
    "ylabel": r"P($\Delta$)",
    "xlabel": r"$\Delta$",
}

info_avg_n_plot = {
    "skip_calculation": False,
    "file_name": "nAvgDict.pkl",
    "vline": {"show": True, "val": n_exp},
    "x_lim": [0, 2],
    "text_size": 15,
    "T": T,
    "bw_list": [3, 3, 0.5, 0.5, 0.5, 0.5, 0.5],
    "save_as": "avg_n",
    "ylabel": r"P(n)",
    "xlabel": "n",
}

info_gap_plot = {
    "skip_calculation": True,
    "T": T,
    "file_name": ["eGapDict.pkl", "deltaGapDict.pkl"],
    "save_as": "energy_gap",
    "save_fig": False,
}

info_dos_plot = {
    "skip_calculation": True,
    "evecs_file_name": "evectors_",  # without seedinfo etc.
    "evals_file_name": "evalues_",
    "alpha_list_dos_plot": [0],
    "v_vals_dos_plot": [0.1],
    "s_fac": 10,
    "temp": T,
    "lloc": ["upper right", "upper right", "lower right"],
    "alpha_text_pos": [-1.5, 0.75],
    "ylabel_pos": [0.005, 0.5],
    "xlabel_pos": [0.5, 0.04],
    "x_lim": [-5, 5],
    "legend_fs": 10,
    "text_fs": 20,
}

info_prob_zero_plot = {
    "skip_calculation": True,
    "T": T,
    "file_name": "deltaDict.pkl",
    "save_as": "prob_zero",
    "factor": 0.1,
}

info_egap_temp_plot = {
    "skip_calculation": True,
    "alpha": 0,
    "save_as": "egap_vs_T",
    "ylabel": r"$E_{gap}$",
    "xlabel": r"T/t",
    "y_lim": [0, 0.2],
    "file_name": "eGapDict.pkl",
}

info_deltagap_temp_plot = {
    "skip_calculation": True,
    "alpha": 0,
    "save_as": "deltagap_vs_T",
    "ylabel": r"$\Delta_{OP}$",
    "xlabel": r"T/t",
    "y_lim": [0, 0.2],
    "file_name": "deltaGapDict.pkl",
}

info_p_uv_dict = {
    "skip_calculation": True,
    "alpha_list": alpha_list_plot,
    "T_list": temp_list,
    "v_vals": v_vals_plot,
    "seed_list": [1, 3],
    "evec_name_prefix": "evectors",
    "file_format": "pkl",
    "save_as": "P_uv.pkl"
}

info_stiffness = {
    "skip_calculation": True,
    "α": 0,
    "T": T,
    "v_vals": v_vals_plot,
    "kx_filename": f"{post_processing_folder}plot_kx.pkl",
    "λxx_filename": f"{post_processing_folder}lambda_dict_full.pkl",
    "save_as": f"{images_folder}stiffness.eps"
}
