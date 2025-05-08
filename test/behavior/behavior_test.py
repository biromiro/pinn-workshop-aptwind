import unittest
import torch
import numpy as np
import numpy.testing as npt
import os, sys

sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'assignment'))
from Burgers import lossfunction as loss_func_burgers
from Burgers import Net as BurgersNet
from utils import prep_data
from Euler import lossfunction as loss_func_euler
from Euler import Net as EulerNet
from torch.autograd import grad


class TestBehavior(unittest.TestCase):
    def test_Burgers_architecture(self):
        model = BurgersNet()
        model.load_state_dict(torch.load(os.getcwd() + "/assignment/my_Burgers_model.pt"))
        model.eval()
        num_params = sum(p.numel() for p in model.parameters())
        if (num_params < 100):
            self.fail("This model is probably too simple for the task, consider adding more hidden layers and nodes.")

    def test_Burgers_loss(self):
        u_ICBC_NN = torch.ones((100, 1)) * 0.5
        u_ICBC = torch.ones((100, 1)) * 0.2
        u_F = torch.ones((10000, 1)) * 0.7
        ut_F = torch.ones((10000, 1)) * 0.3
        ux_F = torch.ones((10000, 1)) * 0.6
        uxx_F = torch.ones((10000, 1)) * 0.9

        expected_loss = np.array([0.6043])
        actual_loss = loss_func_burgers(u_ICBC_NN, u_ICBC, u_F, ut_F, ux_F, uxx_F).numpy()
        npt.assert_array_almost_equal(expected_loss, actual_loss, 4)

    def test_Burgers_performance(self):
        ICBC_POINTS = 100
        F_POINTS = 10000
        # path = "../assignment/burgers_shock.mat"
        path = os.getcwd() + "/assignment/burgers_shock.mat"
        x, t, x_mesh, t_mesh, u_target_mesh, Phi, u_target, \
        Phi_ICBC, u_ICBC, Phi_F = prep_data(path, ICBC_POINTS, F_POINTS)
        Phi = torch.from_numpy(Phi).float()
        Phi_ICBC = torch.from_numpy(Phi_ICBC).float()
        u_ICBC = torch.from_numpy(u_ICBC).float()
        Phi_F = torch.from_numpy(Phi_F).float()
        Phi_F.requires_grad = True
        model = BurgersNet()
        # model.load_state_dict(torch.load("../assignment/my_Burgers_model.pt"))
        model.load_state_dict(torch.load(os.getcwd() + "/assignment/my_Burgers_model.pt"))
        model.eval()
        u_F_NN = model(Phi_F)
        g = grad(u_F_NN.sum(), Phi_F, create_graph=True)[0]
        gg = grad(g[:, 0:1], Phi_F, grad_outputs=torch.ones(u_F_NN.shape), create_graph=True)[0]
        ux_F = g[:, 0:1]
        ut_F = g[:, 1:2]
        uxx_F = gg[:, 0:1]
        u_pred = model(Phi)
        u_ICBC_NN = model(Phi_ICBC)
        loss = loss_func_burgers(u_ICBC_NN, u_ICBC, u_F_NN, ut_F, ux_F, uxx_F)
        L2 = np.linalg.norm((u_pred.detach().numpy() - u_target))
        if (loss.detach().numpy() > 0.005):
            self.fail("Model predictions are not good enough.")

    def test_Euler_architecture(self):
        model = EulerNet()
        model.load_state_dict(torch.load(os.getcwd() + "/assignment/my_1DEuler_model.pt"))
        model.eval()
        num_params = sum(p.numel() for p in model.parameters())
        if (num_params < 100):
            self.fail("This model is probably too simple for the task, consider adding more hidden layers and nodes.")

    def test_Euler_loss(self):
        # tensors = torch.load("./tensors.pt")
        tensors = torch.load(os.getcwd() + "/behavior/tensors.pt")
        F_grads = tensors["F_grads"]
        nabla_rho_tensors = tensors["nabla_rho_tensors"]
        probe_tensors = tensors["probe_tensors"]
        mass0_tensors = tensors["mass0_tensors"]
        expected_loss = np.array([4.6664495])
        actual_loss = loss_func_euler(F_grads, nabla_rho_tensors, probe_tensors, mass0_tensors).detach().numpy()
        npt.assert_array_almost_equal(expected_loss, actual_loss, 4)

    def test_Euler_performance(self):
        path = os.getcwd() + "/assignment/1DEuler_data.npy"
        data = np.load(path, allow_pickle=True)
        F_data, nabla_rho, probe, mass0 = data
        x_F, t_F = F_data[0], F_data[1]
        x_nabla_rho, xdx_nabla_rho, t_nabla_rho, rho_nabla_rho, rhodx_nabla_rho = nabla_rho[0], nabla_rho[1], nabla_rho[
            2], nabla_rho[3], nabla_rho[4]
        x_probe, t_probe, p_probe = probe[0], probe[1], probe[2]
        x_mass0, t_mass0, rho_integral_mass0 = mass0[0], mass0[1], mass0[2]
        Phi_nabla_rho = np.hstack([x_nabla_rho.reshape(-1, 1), t_nabla_rho.reshape(-1, 1)])
        Phidx_nabla_rho = np.hstack([xdx_nabla_rho.reshape(-1, 1), t_nabla_rho.reshape(-1, 1)])
        Phi_probe = np.hstack([x_probe.reshape(-1, 1), t_probe.reshape(-1, 1)])
        Phi_mass0 = np.hstack([x_mass0.reshape(-1, 1), t_mass0.reshape(-1, 1)])
        Phi_F = np.hstack([x_F.reshape(-1, 1), t_F.reshape(-1, 1)])
        Phi_nabla_rho = torch.from_numpy(Phi_nabla_rho).float()
        Phidx_nabla_rho = torch.from_numpy(Phidx_nabla_rho).float()
        rho_nabla_rho = torch.from_numpy(rho_nabla_rho).float()
        rhodx_nabla_rho = torch.from_numpy(rhodx_nabla_rho).float()
        Phi_probe = torch.from_numpy(Phi_probe).float()
        p_probe = torch.from_numpy(p_probe).float()
        Phi_mass0 = torch.from_numpy(Phi_mass0).float()
        rho_integral_mass0 = torch.from_numpy(np.array(rho_integral_mass0)).float()
        Phi_F = torch.from_numpy(Phi_F).float()
        Phi_nabla_rho.requires_grad = True
        Phi_probe.requires_grad = True
        Phi_mass0.requires_grad = True
        Phi_F.requires_grad = True
        model = EulerNet()
        model.load_state_dict(torch.load(os.getcwd() + "/assignment/my_1DEuler_model.pt"))
        model.eval()
        F_pred = model(Phi_F)
        rho_F_NN = F_pred[:, 0]
        u_F_NN = F_pred[:, 1]
        p_F_NN = F_pred[:, 2]
        E_F_NN = (p_F_NN / (rho_F_NN * (1.4 - 1.0))) + (rho_F_NN * torch.square(u_F_NN)) / 2.0
        rho_F_g = grad(rho_F_NN.sum(), Phi_F, create_graph=True)[0]
        u_F_g = grad(u_F_NN.sum(), Phi_F, create_graph=True)[0]
        p_F_g = grad(p_F_NN.sum(), Phi_F, create_graph=True)[0]
        E_F_g = grad(E_F_NN.sum(), Phi_F, create_graph=True)[0]
        rhox_F = rho_F_g[:, 0]
        ux_F = u_F_g[:, 0]
        px_F = p_F_g[:, 0]
        Ex_F = E_F_g[:, 0]
        rhot_F = rho_F_g[:, 1]
        ut_F = u_F_g[:, 1]
        pt_F = p_F_g[:, 1]
        Et_F = E_F_g[:, 1]
        F_grads = [rho_F_NN, u_F_NN, p_F_NN, E_F_NN, rhox_F, ux_F, px_F, Ex_F, rhot_F, ut_F, pt_F, Et_F]
        nabla_rho_pred = model(Phi_nabla_rho)
        dx_nabla_rho_pred = model(Phidx_nabla_rho)
        rho_nabla_rho_NN = nabla_rho_pred[:, 0]
        rhodx_nabla_rho_NN = dx_nabla_rho_pred[:, 0]
        nabla_rho_tensors = [rho_nabla_rho, rhodx_nabla_rho, rho_nabla_rho_NN, rhodx_nabla_rho_NN]
        probe_pred = model(Phi_probe)
        p_probe_NN = probe_pred[:, 2]
        probe_tensors = [p_probe, p_probe_NN]
        mass0_pred = model(Phi_mass0)
        rho_mass0_pred = mass0_pred[:, 0]
        rho_mass0_integral_NN = torch.trapz(rho_mass0_pred, Phi_mass0[:, 0])
        mass0_tensors = [rho_integral_mass0, rho_mass0_integral_NN]
        loss = loss_func_euler(F_grads, nabla_rho_tensors, probe_tensors, mass0_tensors)
        if (loss.detach().numpy() > 0.0005):
            self.fail("Model predictions are not good enough.")

    def test_empty(self):
        pass