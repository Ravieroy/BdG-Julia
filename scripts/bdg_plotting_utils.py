import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from benedict import benedict
from matplotlib.pyplot import cm
from plot_params import *
from pathlib import Path
import os
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# important functions

def load_pkl_file(file_name):
    """
    Loads pickle file
    """
    # load the pickle file.
    with open(file_name, 'rb') as handle:
        dict_ = pickle.load(handle)
    return dict_


def save_pkl_file(object_name, file_name):
    """
    Saves object as pickle file
    """
    with open(file_name, 'wb') as handle:
        pickle.dump(object_name, handle, protocol=pickle.HIGHEST_PROTOCOL)


def collate_data(seed_list, dict_list, results_folder, post_processing_folder):
    """
    Collates data from different files and stores
    it as one single collated file
    """
    Path(post_processing_folder).mkdir(parents=True, exist_ok=True)
    state = info_continued_calc["state"]
    var = info_continued_calc["var"]
    run = info_continued_calc["run"]
    for name in dict_list:
        if state is True:
            name = f"{name}Run{var}{run}"
        new_dict = {}
        for seed in seed_list:
            full_name = os.path.join(results_folder, f"{name}_{seed}.pkl")
            assert os.path.isfile(full_name), f"{full_name} File Not Found"
            dict_ = load_pkl_file(full_name)
            new_dict[seed] = dict_
        new_name = os.path.join(post_processing_folder, f"{name}.pkl")
        save_pkl_file(new_dict, new_name)


def update_collated_dict(info_dict):
    info_dict = benedict(info_dict)
    appending_vals = info_dict.var_values
    state = info_dict.state
    var = info_dict.var
    run = info_dict.run
    dict_list = info_dict.dict_list
    seed_list = info_dict.seed_list
    alpha_list = info_dict.alpha_list
    v_vals = info_dict.v_vals
    collate_data(seed_list, dict_list, results_folder, post_processing_folder)
    for name in dict_list:
        if state is True:
            old_name = name
            new_name = f"{name}Run{var}{run}"
        for seed in seed_list:
            full_old_name = os.path.join(post_processing_folder, f"{old_name}.pkl")
            full_new_name = os.path.join(post_processing_folder, f"{new_name}.pkl")
            assert os.path.isfile(full_old_name), f"{full_old_name} File Not Found"
            assert os.path.isfile(full_new_name), f"{full_new_name} File Not Found"
            dict_old = load_pkl_file(full_old_name)
            dict_new = load_pkl_file(full_new_name)
            if var == "T":
                for alpha in alpha_list:
                    for V in v_vals:
                        for T in appending_vals:
                            dict_old[seed][alpha][V][T] = dict_new[seed][alpha][V][T]
            elif var == "V":
                for alpha in alpha_list:
                    for V in appending_vals:
                        dict_old[seed][alpha][V] = dict_new[seed][alpha][V]
            updated_name = os.path.join(post_processing_folder, f"{old_name}.pkl")
            save_pkl_file(dict_old, updated_name)




def get_concatenated_dict(file_name):
    """
    Collects values from different seed results and makes one big list
    dict[α][V][T]
    """
    assert os.path.isfile(file_name), f"{file_name} File Not Found"
    dict_full = load_pkl_file(file_name)
    dict_ = {}
    for alpha in alpha_list_plot:
        dict_V = {}
        for V in v_vals_plot:
            dict_T = {}
            for T in temp_list:
                list_ = []
                for seed in seed_list:
                    list_tmp = dict_full[seed][alpha][V][T]
                    list_.append(list_tmp)
                # [[1, 2], [3, 4]] => [1, 2, 3, 4]
                full_list = list(np.concatenate(list_).flat)
                dict_T[T] = full_list
            dict_V[V] = dict_T
        dict_[alpha] = dict_V
    return dict_


def plot_probability_distribution(info_dict):
    info_dict = benedict(info_dict)
    file_name = f"{post_processing_folder}{info_dict.file_name}"
    show_vline = info_dict.vline.show
    vline = info_dict.vline.val
    x_lim = info_dict.x_lim
    text_size = info_dict.text_size
    T = info_dict.T
    bw_list = info_dict.bw_list
    name = info_dict.save_as
    ylabel = info_dict.ylabel
    xlabel = info_dict.xlabel

    Path(images_folder).mkdir(parents=True, exist_ok=True)
    assert os.path.isfile(file_name), f"{file_name} File Not Found"
    dict_ = get_concatenated_dict(file_name)

    row = 1
    col = len(alpha_list_plot)
    fig, ax = plt.subplots(row, col, figsize=(8, 5))

    for i in range(len(alpha_list_plot)):
        for j in range(len(v_vals_plot)):
            sns.kdeplot(dict_[alpha_list_plot[i]][v_vals_plot[j]][T], ax=ax,
                        bw_adjust=bw_list[j], label=fr"$V$ = {v_vals_plot[j]}")
            ax.set_xlim(x_lim)
            if show_vline is True:
                ax.axvline(vline, linestyle='-.')
        ax.legend(prop={'size': text_size})
        ax.set_ylabel(ylabel, fontsize=text_size)
        ax.set_xlabel(xlabel, fontsize=text_size)
        ax.tick_params(axis='both', which='major', labelsize=text_size)
    fig.suptitle(f"N = {N}x{N} <n>={n_exp} U={U} T={T}")
    plt.savefig(f"{images_folder}{name}_distribution.png")
    plt.close("all")


def get_avg_egap_dict(file_name):
    """
    Returns a dictionary with quasiparticle gap averaged over seed_list
    dict[α][T][V]
    """
    assert os.path.isfile(file_name), f"{file_name} File Not Found"
    dict_full = load_pkl_file(file_name)
    dict_ = {}
    for alpha in alpha_list_plot:
        egap_T = {}
        for T in temp_list:
            egap_V = {}
            for V in v_vals_plot:
                egap = 0
                for seed in seed_list:
                    egap += dict_full[seed][alpha][V][T]
                avg_egap = egap/len(seed_list)
                egap_V[V] = avg_egap
            egap_T[T] = egap_V
        dict_[alpha] = egap_T
    return dict_


def plot_gap(info_dict):
    info_dict = benedict(info_dict)
    Path(images_folder).mkdir(parents=True, exist_ok=True)

    egap = f"{post_processing_folder}{info_dict.file_name[0]}"
    delta_gap = f"{post_processing_folder}{info_dict.file_name[1]}"

    avg_egap_dict_full = get_avg_egap_dict(egap)
    avg_delta_gap_dict_full = get_avg_egap_dict(delta_gap)

    name = info_dict.save_as
    T = info_dict.T
    for alpha in alpha_list_plot:
        plt.plot(v_vals_plot, list(avg_egap_dict_full[alpha][T].values()),
                 '^-', label=r"$E_{gap}/t$")
        plt.plot(v_vals_plot, list(avg_delta_gap_dict_full[alpha][T].values()),
                 'D-', label=r"$\Delta_{OP}/t$")
    plt.xlabel("V/t")
    plt.title(fr"N={N}, $\langle n \rangle={n_exp}, U={-U}t, T={T}t$")
    plt.legend()
    plt.savefig(f"{images_folder}{name}.png")
    plt.close("all")


def get_dos(alpha_list, v_vals, T, evec_dict, eval_dict, s_fac=20):
    """
    Calculates DOS for given values of alpha and V
    dict_[alpha][V]
    """
    omega_dict = {}
    dos_dict = {}
    for alpha in alpha_list:
        tmp_omega_dict = {}
        tmp_dos_dict = {}
        for V in v_vals:
            u = np.transpose(evec_dict[alpha][V][T])[:n_sites]
            v = np.transpose(evec_dict[alpha][V][T])[n_sites:]

            u = u.T
            v = v.T

            E = eval_dict[alpha][V][T]
            res = DoS(u, v, E, s_fac)
            tmp_omega_dict[V] = res[0]
            tmp_dos_dict[V] = res[1]
        omega_dict[alpha] = tmp_omega_dict
        dos_dict[alpha] = tmp_dos_dict
    return omega_dict, dos_dict


def DoS(u, v, E, s_fac):
    E_max = max(E)
    E_min = min(E)
    E_gap = E[n_sites]

    omega1 = np.linspace(E_min, -E_gap, 200)
    omega2 = np.linspace(E_gap, E_max, 200)

    omega_list = omega1.tolist() + omega2.tolist()
    omega = np.array(omega_list)
    dos = np.zeros(len(omega))
    a = s_fac * (E_max - E_min - 2 * E_gap) / 400

    for w in range(len(omega)):
        if omega[w] == -E_gap or omega[w] == E_gap or omega[w] == E_min or omega[w] == E_max:
            dos[w] = 0
        else:
            sum = 0
            for n in range(len(E)):
                if E[n] >= omega[w]-a/2 and E[n] < omega[w]+a/2:
                    for i in range(n_sites):
                        sum += ((u[n, i]) ** 2 + (v[n, i]) ** 2) / (a * n_sites)
            dos[w] = sum
    return omega, dos


def plot_dos(info_dict):
    print("INFO : Plotting DOS can take some time. Please wait.")

    # initialise benedict and create images folder
    info_dict = benedict(info_dict)
    Path(images_folder).mkdir(parents=True, exist_ok=True)

    seed_plot = np.random.choice(seed_list)

    # load variables from dict
    alpha_list = info_dict.alpha_list_dos_plot
    v_vals = info_dict.v_vals_dos_plot
    T = info_dict.temp

    αx = info_dict.alpha_text_pos[0]  # x position of text α=0
    αy = info_dict.alpha_text_pos[1]

    xlx = info_dict.xlabel_pos[0]  # x label x
    xly = info_dict.xlabel_pos[1]  # x label y

    ylx = info_dict.ylabel_pos[0]  # y label x
    yly = info_dict.ylabel_pos[1]

    x_lim = info_dict.x_lim  # x_lim of plots
    legend_fs = info_dict.legend_fs  # legend fontsize
    text_fs = info_dict.text_fs  # text fontsize
    lloc = info_dict.lloc

    evecs_file_name = f"{results_folder}{info_dict.evecs_file_name}{seed_plot}.pkl"
    evals_file_name = f"{results_folder}{info_dict.evals_file_name}{seed_plot}.pkl"

    evecs_dict = load_pkl_file(evecs_file_name)
    evals_dict = load_pkl_file(evals_file_name)

    # calculate DOS
    omega_dict_full, dos_dict_full = get_dos(
        alpha_list,
        v_vals,
        T,
        evecs_dict,
        evals_dict)

    # Plotting
    n_plots = len(alpha_list) * len(v_vals)
    color = cm.rainbow(np.linspace(0, 1, n_plots))
    fig, ax = plt.subplots(n_plots, 1, figsize=(5, 6), sharex=True, sharey=False)
    count = 0

    for alpha in alpha_list:
        for V in v_vals:
            try:
                ax[count].plot(
                    omega_dict_full[alpha][V],
                    dos_dict_full[alpha][V],
                    label=fr'$V$={V}',
                    c=color[count])
                ax[0].text(αx, αy, fr"$\alpha$ = {alpha}", fontsize=text_fs)
                ax[count].legend(loc=lloc[count], fontsize=legend_fs)
                count += 1
            except TypeError:  # Handles the case when only one V is provided
                ax.plot(
                    omega_dict_full[alpha][V],
                    dos_dict_full[alpha][V],
                    label=fr'$V$={V}',
                    c=color[count])
                ax.text(αx, αy, fr"$\alpha$ = {alpha}", fontsize=text_fs)
                ax.legend(loc=lloc[count], fontsize=legend_fs)

        fig.text(xlx, xly, r'$\omega$', ha='center', fontsize=text_fs)
        fig.text(ylx, yly, r'$N(\omega)$', va='center', rotation='vertical', fontsize=text_fs)
        plt.xlim(x_lim)
        fig.suptitle(f"N = {N}x{N} <n>={n_exp} U={U}")
        plt.savefig(f"{images_folder}DOS_alpha_{alpha}_T_{T}.png")
        plt.close("all")


def get_avg_delta(file_name):
    """
    Returns a dictionary with average value of order parameter Δ
    dict[α][V][T]
    """
    assert os.path.isfile(file_name), f"{file_name} File Not Found"
    delta_dict = get_concatenated_dict(file_name)
    delta_avg_dict = {}
    for alpha in alpha_list_plot:
        delta_avg_dict_V = {}
        for V in v_vals_plot:
            delta_avg_dict_T = {}
            for T in temp_list:
                list_ = delta_dict[alpha][V][T]
                sum_ = np.sum(list_)
                avg = sum_/len(list_)
                delta_avg_dict_T[T] = avg
            delta_avg_dict_V[V] = delta_avg_dict_T
        delta_avg_dict[alpha] = delta_avg_dict_V
    return delta_avg_dict


def plot_probability_zero_delta(info_dict):
    # initialise benedict and get variables
    info_dict = benedict(info_dict)
    factor = info_dict.factor
    name = info_dict.save_as
    T = info_dict.T
    file_name = f"{post_processing_folder}{info_dict.file_name}"
    assert os.path.isfile(file_name), f"{file_name} File Not Found"
    delta_dict = get_concatenated_dict(file_name)
    # calculate avg delta for each α, V and T.
    delta_avg_dict = get_avg_delta(file_name)
    P_delta0_full = []
    for alpha in alpha_list_plot:
        P_delta0 = []
        for V in v_vals_plot:
            list_ = delta_dict[alpha][V][T]
            avg = delta_avg_dict[alpha][V][T]
            threshold = factor * avg
            P = (np.count_nonzero(list_ < threshold))/(len(list_))
            P_delta0.append(P)
        P_delta0_full.append(P_delta0)
    for i in range(len(alpha_list_plot)):
        plt.plot(v_vals_plot, P_delta0_full[i], "--o", label=fr"$\alpha$={alpha_list_plot[i]}")
    plt.xlabel(r"$V$")
    plt.ylabel(r"$P(\Delta = 0)$")
    plt.legend()
    plt.savefig(f"{images_folder}{name}.png")
    plt.close("all")


def plot_gap_vs_temp(info_dict):
    info_dict = benedict(info_dict)
    Path(images_folder).mkdir(parents=True, exist_ok=True)

    alpha = info_dict.alpha
    xlabel = info_dict.xlabel
    ylabel = info_dict.ylabel
    y_lim = info_dict.y_lim
    egap = f"{post_processing_folder}{info_dict.file_name}"

    avg_egap_dict_full = get_avg_egap_dict(egap)
    egap_name = info_dict.save_as
    for V in v_vals_plot:
        val_list = []
        for T in temp_list:
            val = avg_egap_dict_full[alpha][T][V]
            val_list.append(val)
        plt.plot(temp_list, val_list, "-o", label=f"V/t = {V}")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.ylim(y_lim)
    plt.title(fr"N={N}, $\langle n \rangle={n_exp}, U={-U}t$")
    plt.legend()
    plt.savefig(f"{images_folder}{egap_name}.png")
    plt.close("all")


def get_egaps(info_dict):
    info_dict = benedict(info_dict)
    alpha = info_dict.alpha
    egap = f"{post_processing_folder}{info_dict.file_name}"

    avg_egap_dict_full = get_avg_egap_dict(egap)
    gaps_dict = {}
    for V in v_vals_plot:
        val_dict = {}
        for T in temp_list:
            val = avg_egap_dict_full[alpha][T][V]
            val_dict[T] = val
        gaps_dict[V] = val_dict

    for V in v_vals_plot:
        print("----------------------------")
        print(f"V = {V}")
        print("----------------------------")
        for T in temp_list:
            print(f"T: {T} ==> {gaps_dict[V][T]}")

def sort_evec(evec):
    evec_T = evec.T[n_sites:]  # for E>0
    evecs = []
    u_n = []
    v_n = []
    for val in range(n_sites):
        evecs.append(evec_T[val])
        u_n.append(evecs[val][0:int(n_sites)])
        v_n.append(evecs[val][int(n_sites):int(2 * n_sites)])
    return np.array(u_n), np.array(v_n)


def get_p_uv(alpha_list, T_list, v_vals, evector):
    p_uv_dict = {}
    for alpha in alpha_list:
        tmp_p_uv_V = {}
        for V in v_vals:
            tmp_p_uv_T = {}
            for T in T_list:
                un, vn = sort_evec(np.transpose(evector[alpha][V][T]))
                p_uv = []
                for n in range(n_sites):
                    total = 0
                    for r in range(n_sites):
                        total += abs(un[n, r] ** 2 - vn[n, r] ** 2)
                    p_uv.append(total)
                P_uv = np.array(p_uv)
                tmp_p_uv_T[T] = P_uv
            tmp_p_uv_V[V] = tmp_p_uv_T
        p_uv_dict[alpha] = tmp_p_uv_V
    return p_uv_dict


def save_p_uv(info_dict):
    # initialise benedict and get variables
    info_dict = benedict(info_dict)
    alpha_list = info_dict.alpha_list
    T_list = info_dict.T_list
    v_vals = info_dict.v_vals
    seed_list = info_dict.seed_list
    ev_name = info_dict.evec_name_prefix
    file_format = info_dict.file_format
    P_uv_name = info_dict.save_as
    P_uv_dict_full = {}
    for seed in seed_list:
        file_name = f"{results_folder}{ev_name}_{seed}.{file_format}"
        assert os.path.isfile(file_name), f"{file_name} File Not Found"
        evector = load_pkl_file(file_name)
        P_uv_dict = get_p_uv(alpha_list, T_list, v_vals, evector)
        P_uv_dict_full[seed] = P_uv_dict
    P_uv_full_name = f"{results_folder}{P_uv_name}"
    save_pkl_file(P_uv_dict_full, P_uv_full_name)


def plot_lattice_with_coords(df):
    x = df["sitePosX"]
    y = df["sitePosY"]
    n_sites = df.shape[0]
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.scatter(x, y)
    for site in range(n_sites):
        xpos = x[site]
        ypos = y[site]
        pos = (xpos, ypos)
        ax.text(xpos, ypos+0.1, str(pos))
        # ax.text(xpos, ypos-0.1, str((site, site+1)))
        ax.text(xpos, ypos-0.1, str((site)))
        plt.savefig(f"{images_folder}{lattice}.png")


def plot_stiffness(info_dict):
    info_dict = benedict(info_dict)
    kx_filename = info_dict.kx_filename
    λxx_filename = info_dict.λxx_filename
    filename = info_dict.save_as
    Path(images_folder).mkdir(parents=True, exist_ok=True)

    assert os.path.isfile(kx_filename), f"{kx_filename} File Not Found"
    assert os.path.isfile(λxx_filename), f"{λxx_filename} File Not Found"
    kx_list = load_pkl_file(kx_filename)
    λ_dict = load_pkl_file(λxx_filename)

    lambda_avg_list = []
    for V in v_vals:
        sum_ = 0
        for seed in seed_list:
            sum_ += np.sum(λ_dict[seed][V])
        lambda_avg_list.append(sum_/(len(seed_list) * n_sites))
    plt.plot(kx_list[0], kx_list[1], '--o')
    plt.plot(v_vals, -np.array(lambda_avg_list), '--o')
    plt.savefig(f"{images_folder}{filename}")
    plt.clf()
    plt.plot(v_vals, kx_list[1] + np.array(lambda_avg_list), '--o')
    plt.savefig(f"{images_folder}Ds_plot.png")
