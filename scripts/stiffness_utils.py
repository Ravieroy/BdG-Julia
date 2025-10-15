import numpy as np
from numba import jit
import pandas as pd
from bdg_plotting_utils import (
    save_pkl_file,
    load_pkl_file,
)
from plot_params import *


def find_index_in_df(pos, df):
    for site in range(n_sites):
        xpos = df.iloc[site]["sitePosX"]
        ypos = df.iloc[site]["sitePosY"]
        if xpos == pos[0] and ypos == pos[1]:
            index = site
            break
        else:
            index = -1
    return index


def get_rhs_site(df):
    ex = np.array([0.5, 0])
    sites = np.arange(0, N, 0.25)
    rightSite = {}
    df = pd.read_csv(df)
    for i in range(n_sites):
        site_pos_X = df.iloc[i]["sitePosX"]
        site_pos_Y = df.iloc[i]["sitePosY"]
        coord = np.array([site_pos_X, site_pos_Y])
        layer = int(site_pos_Y/0.433)
        if site_pos_Y / 0.433 % 2 == 0:  # Atom A and B
            iplusx = coord + ex
            if find_index_in_df(iplusx, df) != -1:
                rIndex = find_index_in_df(iplusx, df)
                rightSite[i] = rIndex
            else:  # PBC
                iplusx = [sites[layer], site_pos_Y]
                rIndex = find_index_in_df(iplusx, df)
                rightSite[i] = rIndex
        else:  # Atom C
            iplusx = coord + 2 * ex
            if find_index_in_df(iplusx, df) != -1:
                rIndex = find_index_in_df(iplusx, df)
                rightSite[i] = rIndex
            else:  # PBC
                iplusx = [sites[layer], site_pos_Y]
                rIndex = find_index_in_df(iplusx, df)
                rightSite[i] = rIndex
    return rightSite


@jit(nopython=True)
def calc_kx(right_site_list, v, n_sites, t=1):
    sum_ = 0
    for n in range(n_sites):
        for r in range(n_sites):
            rplusx = right_site_list[r]
            sum_ += v[n, r] * v[n, rplusx]
    return (4 * t * sum_)/n_sites


def sort_evec(evec):
    evec_T = evec.T[n_sites:]  # for E >0
    evecs = []
    u_n = []
    v_n = []
    for val in range(n_sites):
        evecs.append(evec_T[val])
        u_n.append(evecs[val][0:int(n_sites)])
        v_n.append(evecs[val][int(n_sites):int(2 * n_sites)])
    return np.array(u_n), np.array(v_n)


def initialize_empty_dict(name):
    dict_ = {}
    save_pkl_file(dict_, name)


def append_dict(dict_, key, name):
    full_dict = load_pkl_file(name)
    full_dict[key] = dict_
    save_pkl_file(full_dict, name)


@jit(nopython=True)
def calc_lambda_xx_rij(right_site_list, nmax, r1, r2, u, v, e_val, t):
    n_sites = 3 * nmax ** 2
    r1_ex = right_site_list[r1]
    r2_ex = right_site_list[r2]
    sum_ = 0
    for n in range(n_sites):
        for m in range(n_sites):
            En = e_val[n]
            Em = e_val[m]
            sum_ += ((1/(En + Em))*(v[m, r2_ex]*u[n, r2] + v[n, r2_ex]*u[m, r2])*(u[n, r1_ex]*v[m, r1] + v[n, r1]*u[m, r1_ex] - u[n, r1]*v[m, r1_ex] - v[n,r1_ex]*u[m,r1]))
            sum_ += ((1/(En + Em))*(u[m, r2_ex]*v[n, r2] + u[n, r2_ex]*v[m, r2])*(v[n, r1_ex]*u[m, r1] + u[n, r1]*v[m, r1_ex] - v[n, r1]*u[m, r1_ex] - u[n,r1_ex]*v[m,r1]))
    return 2 * (t ** 2) * sum_
