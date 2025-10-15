import numpy as np
import matplotlib.pyplot as plt
import time

from bdg_plotting_utils import save_pkl_file, load_pkl_file
from stiffness_utils import sort_evec, get_rhs_site, calc_kx

from plot_params import (
    df_filename,
    info_stiffness,
    seed_list,
    t,
    N,
    results_folder,
    images_folder
)

start = time.time()
df = df_filename
α = info_stiffness["α"]
v_vals = info_stiffness["v_vals"]
T = info_stiffness["T"]
kx_filename = info_stiffness["kx_filename"]
# λxx_filename = info_stiffness["λxx_filename"]
right_site_dict = get_rhs_site(df)
rMat = np.array(list(right_site_dict.values()))

kx_val_avg = []
for V in v_vals:
    sum_ = 0
    for seed in seed_list:
        evec_dict = load_pkl_file(f"{results_folder}/evectors_{seed}.pkl")
        ev_transpose = np.transpose(evec_dict[α][V][T])
        u, v = sort_evec(ev_transpose)
        kx_val = calc_kx(rMat, v, N, t)
        sum_ += kx_val
    kx_val_avg.append(sum_/len(seed_list))


plot_arr = [v_vals, kx_val_avg]
save_pkl_file(plot_arr, kx_filename)
kx_av = np.array(kx_val_avg)
plt.plot(v_vals, kx_av, '--o')
plt.savefig(f"{images_folder}kx.png")
print("Calculation for kx done")
end = time.time()
print("time taken", end - start, 's')
