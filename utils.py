import torch
import torch.nn as nn
from torch.autograd import grad

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from IPython.display import HTML
from tqdm.notebook import tqdm, trange
from clawpack.riemann.euler_1D_py import euler_hllc_1D
import numpy as np
import scipy.io
import os
import sys

def lhs(n, samples=None):
    """
    Generate a Latin-hypercube design (each factor uniform in [0,1]).
    Args:
        n (int): Number of dimensions (factors).
        samples (int, optional): Number of samples. Defaults to n.
    Returns:
        np.ndarray: Array of shape (samples, n).
    """
    if samples is None:
        samples = n
    cut = np.linspace(0, 1, samples + 1)
    # low and high edges for each of the samples
    a = cut[:samples]
    b = cut[1:samples+1]

    u = np.random.rand(samples, n)
    rd = np.zeros_like(u)
    for j in range(n):
        rd[:, j] = u[:, j] * (b - a) + a

    H = np.zeros_like(rd)
    for j in range(n):
        idx = np.random.permutation(samples)
        H[:, j] = rd[idx, j]

    return H


def load_data_burgers(path, icbc_size, f_size):
    """
    Load Burgers shock data and sample IC/BC and collocation points.
    Returns x, t, mesh X, mesh T, u_target_mesh, Phi_icbc, u_icbc, Phi_f.
    """
    data = scipy.io.loadmat(path)
    # Flatten to column vectors
    t = data['t'].flatten()[:, None]
    x = data['x'].flatten()[:, None]
    u_mesh = np.real(data['usol']).T  # time x space
    # Meshgrid for plotting
    X, T = np.meshgrid(x, t)
    # Full grid points
    Phi_all = np.hstack([X.reshape(-1,1), T.reshape(-1,1)])
    u_all = u_mesh.flatten()[:, None]
    # IC/BC: t=0, x=-1, x=1
    xb1 = np.hstack([X[0:1].T, T[0:1].T])  # initial
    ub1 = u_mesh[0:1].T
    xb2 = np.hstack([X[:,0:1], T[:,0:1]])  # left boundary
    ub2 = u_mesh[:,0:1]
    xb3 = np.hstack([X[:,-1:], T[:,-1:]])  # right boundary
    ub3 = u_mesh[:,-1:]
    X_icbc = np.vstack([xb1, xb2, xb3])
    u_icbc = np.vstack([ub1, ub2, ub3])
    # Subsample IC/BC for training
    idx_icbc = np.random.choice(X_icbc.shape[0], icbc_size, replace=False)
    Phi_icbc = torch.from_numpy(X_icbc[idx_icbc]).float()
    u_icbc = torch.from_numpy(u_icbc[idx_icbc]).float()
    # Collocation points via LHS
    lb = Phi_all.min(axis=0)
    ub = Phi_all.max(axis=0)
    X_f = lb + (ub - lb) * lhs(2, f_size)
    Phi_f = torch.from_numpy(X_f).float()
    return x, t, X, T, u_mesh, Phi_icbc, u_icbc, Phi_f

def load_data_euler(path: str = "./1DEuler_data.npy", device: torch.device = "cpu"):
    """
    Loads and preprocesses 1D Euler training data from a NumPy file.

    Returns:
        data: dict of torch.Tensors (requires_grad set appropriately)
    """
    raw = np.load(path, allow_pickle=True)
    F_data, nabla_rho, probe, mass0 = raw

    # Unpack
    x_F, t_F = F_data[0], F_data[1]
    x_nr, xdx_nr, t_nr, rho_nr, rhodx_nr = nabla_rho
    x_p, t_p, p_p = probe
    x_m0, t_m0, rho_int = mass0

    # Stack inputs
    def to_tensor(x, t):
        pts = np.stack([x.flatten(), t.flatten()], axis=1)
        return torch.tensor(pts, dtype=torch.float32, device=device)

    data = {}
    data['Phi_F'] = to_tensor(x_F, t_F).requires_grad_(True)
    data['True_F'] = {
        'rho': torch.tensor(F_data[2], dtype=torch.float32, device=device) if len(F_data) > 2 else None
    }
    data['Phi_nabla'] = to_tensor(x_nr, t_nr).requires_grad_(True)
    data['Phi_nabla_dx'] = to_tensor(xdx_nr, t_nr).requires_grad_(True)
    data['rho_nr'] = torch.tensor(rho_nr, dtype=torch.float32, device=device)
    data['rhodx_nr'] = torch.tensor(rhodx_nr, dtype=torch.float32, device=device)
    data['Phi_probe'] = to_tensor(x_p, t_p).requires_grad_(True)
    data['p_probe'] = torch.tensor(p_p, dtype=torch.float32, device=device)
    data['Phi_mass0'] = to_tensor(x_m0, t_m0).requires_grad_(True)
    data['rho_int0'] = torch.tensor(rho_int, dtype=torch.float32, device=device)
    return data

def sod_hllc_solution(x, t, left, right, gamma=1.4, x0=0.0):
    """
    Approximate Sod-tube via HLLC at (t,x).
    Returns rho,u,p as shape (len(t),len(x)).
    """
    rho_l,u_l,p_l = left
    rho_r,u_r,p_r = right
    # build conserved:
    E_l = p_l/(gamma-1) + 0.5*rho_l*u_l**2
    E_r = p_r/(gamma-1) + 0.5*rho_r*u_r**2

    # pack into 3×1 for solver:
    ql = np.array([[rho_l, rho_l*u_l, E_l]]).T
    qr = np.array([[rho_r, rho_r*u_r, E_r]]).T
    aux = None
    prob = {'gamma':gamma, 'gamma1':1-gamma}

    nt, nx = len(t), len(x)
    rho = np.zeros((nt,nx)); u = np.zeros_like(rho); p = np.zeros_like(rho)

    for i, ti in enumerate(t):
        xi = (x - x0)/ (ti if ti>0 else 1e-8)
        # repeat ql, qr across all xi:
        Ql = np.repeat(ql[:,:,None], nx, axis=2)
        Qr = np.repeat(qr[:,:,None], nx, axis=2)
        wave, s, amdq, apdq = euler_hllc_1D(Ql, Qr, aux, aux, prob)
        # wave[:,0,:] = left‐star, wave[:,1,:] = right‐star
        s_l, s_m, s_r = s
        for j, zij in enumerate(xi):
            if zij < s_l[j]:
                r,u_,pp = rho_l, u_l, p_l
            elif zij < s_m[j]:
                q = wave[:,0,j] + Ql[:,0,j]
                r,u_,pp = q[0], q[1]/q[0], (gamma-1)*(q[2] - 0.5*q[1]**2/q[0])
            elif zij < s_r[j]:
                q = wave[:,1,j] + wave[:,0,j] + Ql[:,0,j]
                r,u_,pp = q[0], q[1]/q[0], (gamma-1)*(q[2] - 0.5*q[1]**2/q[0])
            else:
                r,u_,pp = rho_r, u_r, p_r
            rho[i,j], u[i,j], p[i,j] = r, u_, pp

    return rho, u, p
