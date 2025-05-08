###########################################################################
#               Physics-Informed Machine Learning                         #
#                             SS 2023                                     #
#                                                                         #
#                     Exercise 8 - Solution                               #
#                                                                         #
# NOTICE: Sharing and distribution of any course content, other than      #
# between individual students registered in the course, is not permitted  #
# without permission.                                                     #
#                                                                         #
###########################################################################

import os
import torch
import matplotlib.pyplot as plt
import torch.nn as nn
from torch.optim import Adam, LBFGS
from torch.autograd import grad
from torch.optim.lr_scheduler import StepLR
import numpy as np

DX = 0.0999


class Net(nn.Module):
    def __init__(self):
        super().__init__()

        hidden_nodes = 120
        self.fc1 = nn.Linear(2, hidden_nodes)
        self.fc2 = nn.Linear(hidden_nodes, hidden_nodes)
        self.fc3 = nn.Linear(hidden_nodes, hidden_nodes)
        self.fc4 = nn.Linear(hidden_nodes, hidden_nodes)
        self.fc5 = nn.Linear(hidden_nodes, 3)

    def forward(self, x):
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        x = torch.tanh(self.fc3(x))
        x = torch.tanh(self.fc4(x))
        x = self.fc5(x)  # not using an activation function gives better results
        return x


def lossfunction(F_grads, nabla_rho_tensors, probe_tensors, mass0_tensors):
    """
        Parameters
        ----------
        F_grads : list of torch.Tensor objects of size F_POINTS
        nabla_rho_tensors : list of torch.Tensor objects of size nabla_rho_POINTS
        probe_tensors : list of torch.Tensor objects of size p_probe_POINTS
        mass0_tensors : list of torch.Tensor objects of size mass0_POINTS

        Returns
        -------
        loss = MSE_F * w_F + MSE_nabla_rho * w_nabla_rho + MSE_probe * w_probe + MSE_mass0 * w_mass0
        """
    rho_F, u_F, p_F, E_F, \
    rhox_F, ux_F, px_F, Ex_F, \
    rhot_F, ut_F, pt_F, Et_F = \
        F_grads[0], F_grads[1], F_grads[2], F_grads[3], \
        F_grads[4], F_grads[5], F_grads[6], F_grads[7], \
        F_grads[8], F_grads[9], F_grads[10], F_grads[11]

    rho_nabla_rho, rhodx_nabla_rho, rho_nabla_rho_NN, rhodx_nabla_rho_NN, dx = \
        nabla_rho_tensors[0], nabla_rho_tensors[1], nabla_rho_tensors[2], nabla_rho_tensors[3], DX

    p_probe, p_probe_NN = probe_tensors[0], probe_tensors[1]

    rho_integral_mass0, rho_mass0_integral_NN = mass0_tensors[0], mass0_tensors[1]

    w_nabla_rho = 1
    w_probe = 1
    w_mass0 = 1
    w_F = 1

    MSE_nabla_rho = None
    MSE_probe = None
    MSE_mass0 = None
    MSE_F = None

    MSE_nabla_rho = torch.sum(
        torch.square(((rhodx_nabla_rho_NN - rho_nabla_rho_NN) / dx) - (rhodx_nabla_rho - rho_nabla_rho) / dx)) / \
                    rhodx_nabla_rho.shape[0]

    MSE_probe = torch.sum(torch.square(p_probe - p_probe_NN)) / p_probe_NN.shape[0]

    MSE_mass0 = torch.sum(torch.square(rho_mass0_integral_NN - rho_integral_mass0))

    MSE_F1 = rhot_F + torch.mul(rho_F, ux_F) + torch.mul(rhox_F, u_F)
    MSE_F2 = torch.mul(rhot_F, u_F) + torch.mul(rho_F, ut_F) + torch.mul(rho_F,
                                                                         2 * ux_F * u_F) + rhox_F * u_F ** 2 + px_F
    MSE_F3 = torch.mul(rhot_F, E_F) + torch.mul(rho_F, Et_F) + \
             torch.mul(ux_F, torch.mul(rho_F, E_F)) + \
             torch.mul(u_F, torch.mul(rhox_F, E_F)) + \
             torch.mul(u_F, torch.mul(rho_F, Ex_F)) + \
             torch.mul(ux_F, p_F) + torch.mul(u_F, px_F)

    MSE_F = torch.sum(torch.square(MSE_F1)) / MSE_F1.shape[0] \
            + torch.sum(torch.square(MSE_F2)) / MSE_F2.shape[0] \
            + torch.sum(torch.square(MSE_F3)) / MSE_F3.shape[0]

    # print("MSE_nabla_rho: %.5f, MSE_probe: %.5f, MSE_mass0: %.5f, MSE_F:%.5f" % (
    #     MSE_nabla_rho, MSE_probe, MSE_mass0, MSE_F))

    loss = MSE_F * w_F + MSE_nabla_rho * w_nabla_rho + MSE_probe * w_probe + MSE_mass0 * w_mass0
    return loss


def train_model(path="./1DEuler_data.npy"):
    # setting a seed for pytorch as well as one for numpy
    torch.manual_seed(2)
    np.random.seed(2)

    # hyperparameters
    F_POINTS = 2000
    nabla_rho_POINTS = 500
    p_probe_POINTS = 200
    mass0_points = 500
    NUM_EPOCHS_ADAM = 3000
    NUM_EPOCHS_LBFGS = 200
    LEARNING_RATE = 0.001

    # generating 1D Euler data
    # data = generate_1D_Euler_data(F_points=F_POINTS, nabla_rho_points=nabla_rho_POINTS,
    #                                                          p_probe_points=p_probe_POINTS, mass0_points=mass0_points)
    data = np.load(path, allow_pickle=True)
    F_data, nabla_rho, probe, mass0 = data
    # unpacking the generated data from lists into numpy arrays
    x_F, t_F = F_data[0], F_data[1]
    x_nabla_rho, xdx_nabla_rho, t_nabla_rho, rho_nabla_rho, rhodx_nabla_rho = nabla_rho[0], nabla_rho[1], nabla_rho[2], \
                                                                              nabla_rho[3], nabla_rho[4]
    x_probe, t_probe, p_probe = probe[0], probe[1], probe[2]
    x_mass0, t_mass0, rho_integral_mass0 = mass0[0], mass0[1], mass0[2]

    # stacking together the x and t inputs
    Phi_nabla_rho = np.hstack([x_nabla_rho.reshape(-1, 1), t_nabla_rho.reshape(-1, 1)])
    Phidx_nabla_rho = np.hstack([xdx_nabla_rho.reshape(-1, 1), t_nabla_rho.reshape(-1, 1)])
    Phi_probe = np.hstack([x_probe.reshape(-1, 1), t_probe.reshape(-1, 1)])
    Phi_mass0 = np.hstack([x_mass0.reshape(-1, 1), t_mass0.reshape(-1, 1)])
    Phi_F = np.hstack([x_F.reshape(-1, 1), t_F.reshape(-1, 1)])

    # transforming the numpy-arrays into torch.Tensors
    Phi_nabla_rho = torch.from_numpy(Phi_nabla_rho).float()
    Phidx_nabla_rho = torch.from_numpy(Phidx_nabla_rho).float()
    rho_nabla_rho = torch.from_numpy(rho_nabla_rho).float()
    rhodx_nabla_rho = torch.from_numpy(rhodx_nabla_rho).float()
    Phi_probe = torch.from_numpy(Phi_probe).float()
    p_probe = torch.from_numpy(p_probe).float()
    Phi_mass0 = torch.from_numpy(Phi_mass0).float()
    rho_integral_mass0 = torch.from_numpy(np.array(rho_integral_mass0)).float()
    Phi_F = torch.from_numpy(Phi_F).float()

    # ensuring that we can get gradients w.r.t. the input data
    Phi_nabla_rho.requires_grad = True
    Phi_probe.requires_grad = True
    Phi_mass0.requires_grad = True
    Phi_F.requires_grad = True

    # creating an instance of our neural network class
    model = Net()

    # creating an array for storing the loss values
    losses = np.zeros(NUM_EPOCHS_ADAM + NUM_EPOCHS_LBFGS)

    # creating an instance of the Adam optimizer
    optimizer = Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    # scheduler = StepLR(optimizer, step_size=2000, gamma=0.5)

    # Adam training loop
    for epoch in range(1, NUM_EPOCHS_ADAM + 1):
        # making a prediction for rho, u, p at the collocation points with the F training data
        F_pred = model(Phi_F)
        rho_F_NN = F_pred[:, 0]
        u_F_NN = F_pred[:, 1]
        p_F_NN = F_pred[:, 2]
        E_F_NN = (p_F_NN / (rho_F_NN * (1.4 - 1.0))) + (rho_F_NN * torch.square(u_F_NN)) / 2.0

        # computing the first derivatives of rho, u, p, E w.r.t. x and t by backpropagating through the network
        rho_F_g = grad(rho_F_NN.sum(), Phi_F, create_graph=True)[0]
        u_F_g = grad(u_F_NN.sum(), Phi_F, create_graph=True)[0]
        p_F_g = grad(p_F_NN.sum(), Phi_F, create_graph=True)[0]
        E_F_g = grad(E_F_NN.sum(), Phi_F, create_graph=True)[0]

        # These are our derivatives w.r.t. x and t
        rhox_F = rho_F_g[:, 0]
        ux_F = u_F_g[:, 0]
        px_F = p_F_g[:, 0]
        Ex_F = E_F_g[:, 0]

        rhot_F = rho_F_g[:, 1]
        ut_F = u_F_g[:, 1]
        pt_F = p_F_g[:, 1]
        Et_F = E_F_g[:, 1]

        # organizing the predictions and derivatives in a list
        F_grads = [rho_F_NN, u_F_NN, p_F_NN, E_F_NN, rhox_F, ux_F, px_F, Ex_F, rhot_F, ut_F, pt_F, Et_F]

        # making a prediction with the nabla_rho training data
        nabla_rho_pred = model(Phi_nabla_rho)
        # making a prediction with the nabla_rho+dx training data
        dx_nabla_rho_pred = model(Phidx_nabla_rho)
        # taking only the rho predictions
        rho_nabla_rho_NN = nabla_rho_pred[:, 0]
        rhodx_nabla_rho_NN = dx_nabla_rho_pred[:, 0]
        # organizing the predictions and values from the training into a list
        nabla_rho_tensors = [rho_nabla_rho, rhodx_nabla_rho, rho_nabla_rho_NN, rhodx_nabla_rho_NN]

        # making a prediction with the pressure_probe training data
        probe_pred = model(Phi_probe)
        # taking only the p predictions
        p_probe_NN = probe_pred[:, 2]
        # organizing the predictions and values from the training into a list
        probe_tensors = [p_probe, p_probe_NN]

        # making a prediction with the mass0 training data
        mass0_pred = model(Phi_mass0)
        # taking only the rho predictions
        rho_mass0_pred = mass0_pred[:, 0]
        # computing the integral over the predicted rho values
        rho_mass0_integral_NN = torch.trapz(rho_mass0_pred, Phi_mass0[:, 0])
        # organizing the true integral and predicted integral into a list
        mass0_tensors = [rho_integral_mass0, rho_mass0_integral_NN]

        optimizer.zero_grad()
        loss = lossfunction(F_grads, nabla_rho_tensors, probe_tensors, mass0_tensors)
        loss.backward()
        optimizer.step()
        # scheduler.step()
        losses[epoch - 1] = loss.detach().numpy()

        # printing the loss at every 100th epoch
        if epoch % 100 == 0:
            print("Epoch: %d, Loss: %.7f" % (epoch, losses[epoch - 1]))

    optimizer = LBFGS(model.parameters(), lr=1, history_size=50)
    # scheduler = StepLR(optimizer, step_size=500, gamma=0.75)

    # LBFGS training loop
    for epoch in range(NUM_EPOCHS_ADAM + 1, NUM_EPOCHS_ADAM + NUM_EPOCHS_LBFGS + 1):
        def closure():
            optimizer.zero_grad()
            # making a prediction for rho, u, p at the collocation points with the F training data
            F_pred = model(Phi_F)
            rho_F_NN = F_pred[:, 0]
            u_F_NN = F_pred[:, 1]
            p_F_NN = F_pred[:, 2]
            E_F_NN = (p_F_NN / (rho_F_NN * (1.4 - 1.0))) + (rho_F_NN * torch.square(u_F_NN)) / 2.0

            # computing the first derivatives of rho, u, p, E w.r.t. x and t by backpropagating through the network
            rho_F_g = grad(rho_F_NN.sum(), Phi_F, create_graph=True)[0]
            u_F_g = grad(u_F_NN.sum(), Phi_F, create_graph=True)[0]
            p_F_g = grad(p_F_NN.sum(), Phi_F, create_graph=True)[0]
            E_F_g = grad(E_F_NN.sum(), Phi_F, create_graph=True)[0]

            # These are our derivatives w.r.t. x and t
            rhox_F = rho_F_g[:, 0]
            ux_F = u_F_g[:, 0]
            px_F = p_F_g[:, 0]
            Ex_F = E_F_g[:, 0]

            rhot_F = rho_F_g[:, 1]
            ut_F = u_F_g[:, 1]
            pt_F = p_F_g[:, 1]
            Et_F = E_F_g[:, 1]

            # organizing the predictions and derivatives in a list
            F_grads = [rho_F_NN, u_F_NN, p_F_NN, E_F_NN, rhox_F, ux_F, px_F, Ex_F, rhot_F, ut_F, pt_F, Et_F]

            # making a prediction with the nabla_rho training data
            nabla_rho_pred = model(Phi_nabla_rho)
            # making a prediction with the nabla_rho+dx training data
            dx_nabla_rho_pred = model(Phidx_nabla_rho)
            # taking only the rho predictions
            rho_nabla_rho_NN = nabla_rho_pred[:, 0]
            rhodx_nabla_rho_NN = dx_nabla_rho_pred[:, 0]
            # organizing the predictions and values from the training into a list
            nabla_rho_tensors = [rho_nabla_rho, rhodx_nabla_rho, rho_nabla_rho_NN, rhodx_nabla_rho_NN]

            # making a prediction with the pressure_probe training data
            probe_pred = model(Phi_probe)
            # taking only the p predictions
            p_probe_NN = probe_pred[:, 2]
            # organizing the predictions and values from the training into a list
            probe_tensors = [p_probe, p_probe_NN]

            # making a prediction with the mass0 training data
            mass0_pred = model(Phi_mass0)
            # taking only the rho predictions
            rho_mass0_pred = mass0_pred[:, 0]
            # computing the integral over the predicted rho values
            rho_mass0_integral_NN = torch.trapz(rho_mass0_pred, Phi_mass0[:, 0])
            # organizing the true integral and predicted integral into a list
            mass0_tensors = [rho_integral_mass0, rho_mass0_integral_NN]

            loss = lossfunction(F_grads, nabla_rho_tensors, probe_tensors, mass0_tensors)
            loss.backward()
            losses[epoch - 1] = loss.detach().numpy()

            return loss

        optimizer.step(closure)
        # scheduler.step()

        # printing the loss at every 100th epoch
        if epoch % 100 == 0:
            print("Epoch: %d, Loss: %.10f" % (epoch, losses[epoch - 1]))

    torch.save(model.state_dict(), "my_1DEuler_model.pt")

    # plotting the training loss over epochs
    plt.figure()
    plt.semilogy(losses)
    plt.title("Training loss")
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.show()

    VAL_POINTS = 500
    # generating a "validation" grid on which we will solve the 1D Euler problem with a neural network
    x_grid = np.arange(-1, 1, 2 / VAL_POINTS)
    t_grid = np.arange(0, 2, 2 / VAL_POINTS)
    x_mesh, t_mesh = np.meshgrid(x_grid, t_grid)

    # organizing the input data and creating a torch.Tensor
    Phi_grid = np.concatenate([x_mesh.reshape(-1, 1), t_mesh.reshape(-1, 1)], axis=1)
    Phi_grid = torch.tensor(Phi_grid).float()

    # computing the predictions and transforming them into numpy arrays
    output = model(Phi_grid)
    output = output.detach().numpy()
    rho_pred = output[:, 0]
    u_pred = output[:, 1]
    p_pred = output[:, 2]

    # computing the exact solution on the same grid
    rho_solution = 1.0 + 0.2 * np.sin(np.pi * (x_mesh.flatten() - t_mesh.flatten()))
    u_solution = np.ones_like(rho_solution)
    p_solution = np.ones_like(rho_solution)

    # plotting the exact solution and predictions
    fig, ax = plt.subplots(2, 3)
    fig.set_size_inches(20, 10)
    ax[0, 0].imshow(rho_solution.reshape(VAL_POINTS, VAL_POINTS), aspect='equal', vmin=0.0, vmax=1.2)
    ax[0, 1].imshow(u_solution.reshape(VAL_POINTS, VAL_POINTS), aspect='equal', vmin=0.0, vmax=1.2)
    ax[0, 2].imshow(p_solution.reshape(VAL_POINTS, VAL_POINTS), aspect='equal', vmin=0.0, vmax=1.2)
    ax[1, 0].imshow(rho_pred.reshape(VAL_POINTS, VAL_POINTS), aspect='equal', vmin=0.0, vmax=1.2)
    ax[1, 1].imshow(u_pred.reshape(VAL_POINTS, VAL_POINTS), aspect='equal', vmin=0.0, vmax=1.2)
    ax[1, 2].imshow(p_pred.reshape(VAL_POINTS, VAL_POINTS), aspect='equal', vmin=0.0, vmax=1.2)
    ax[0, 0].set_title("rho")
    ax[0, 1].set_title("u")
    ax[0, 2].set_title("p")
    ax[1, 0].set_title("rho_NN")
    ax[1, 1].set_title("u_NN")
    ax[1, 2].set_title("p_NN")
    plt.show()

    # l2_density = np.mean((rho_solution - rho_pred) ** 2)
    # print('l2 density', l2_density)

    # computing the L2 error on the val grid
    L2_val = np.mean((rho_solution - rho_pred) ** 2) \
             + np.mean((u_solution - u_pred) ** 2) \
             + np.mean((p_solution - p_pred) ** 2)
    print("L2 error on val grid: %.6f" % L2_val)


def main():
    train_model()


if __name__ == "__main__":
    main()
