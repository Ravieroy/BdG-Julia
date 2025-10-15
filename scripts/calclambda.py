import numpy as np
import time

from plot_params import (
    df_filename,
    info_stiffness,
    seed_list,
    t,
    N,
    n_sites,
    results_folder,
    interval,
    logFolder,
)


from stiffness_utils import (
    sort_evec,
    get_rhs_site,
    calc_lambda_xx_rij,
    initialize_empty_dict,
    append_dict,
)

from bdg_plotting_utils import load_pkl_file
from ray.util.multiprocessing import Pool
# from multiprocessing import Pool


df = df_filename
α = info_stiffness["α"]
v_vals = info_stiffness["v_vals"]
T = info_stiffness["T"]
kx_filename = info_stiffness["kx_filename"]
λxx_filename = info_stiffness["λxx_filename"]
right_site_dict = get_rhs_site(df)
rMat = np.array(list(right_site_dict.values()))


def main(seed):
    # rMat = load_pkl_file(pkl_filename)
    evec_dict = load_pkl_file(f"{results_folder}/evectors_{seed}.pkl")
    eval_dict = load_pkl_file(f"{results_folder}/evalues_{seed}.pkl")
    eta = 0
    n_total = n_sites ** 2
    count = 0
    lambda_dict_V = {}
    for V in v_vals:
        count += 1
        e_val = eval_dict[α][V][T][n_sites:]
        ev_transpose = np.transpose(evec_dict[α][V][T])
        u, v = sort_evec(ev_transpose)
        old_time = time.time()
        lambda_val = np.zeros((n_sites, n_sites))
        count_old = np.count_nonzero(lambda_val)
        for i in range(n_sites):
            for j in range(n_sites):
                lambda_val[i, j] = calc_lambda_xx_rij(rMat, N, i, j, u, v, e_val, t)
                if time.time() - old_time >= interval:
                    #  ------ETA calculation block : BEGIN--------------------
                    present_count = np.count_nonzero(lambda_val)
                    present_time = time.time()
                    dt = present_time - old_time
                    remaining_entries = n_total - present_count
                    diff = present_count - count_old
                    rate = diff/dt
                    eta = (remaining_entries/rate)/60  # in mins
                    count_old = present_count
                    old_time = present_time
                    #  -------------END--------------------------------#
                percent = (n_sites ** 2 - np.count_nonzero(lambda_val))/(n_sites ** 2) * 100
                with open(f"{logFolder}/status_{seed}.log", "w") as f:
                    f.write(f"Remaining : {percent:.2f}% (V={V}({count}/{len(v_vals)}))| Estimated Time Left : {eta:.2f} mins ({eta/60:.2f} hrs)")
        lambda_dict_V[V] = lambda_val
    append_dict(lambda_dict_V, seed, λxx_filename)


# main(1)

if __name__ == "__main__":
    # Initialisation of dictionaries
    initialize_empty_dict(λxx_filename)
    rMat = np.array(list(right_site_dict.values()))
    with Pool() as pool:
        start = time.time()
        result = pool.map(main, seed_list)
        end = time.time()
        print("time taken", end - start, 's')
        pool.close()
