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

import torch
import matplotlib.pyplot as plt
import torch.nn as nn
from utils import prep_data
from torch.optim import Adam
from torch.autograd import grad
from torch.optim.lr_scheduler import StepLR
import numpy as np

class Net(nn.Module):
    def __init__(self):
        super().__init__()

        self.fc1 = nn.Linear(2, 20)
        self.fc2 = nn.Linear(20, 20)
        self.fc3 = nn.Linear(20, 20)
        self.fc4 = nn.Linear(20, 20)
        self.fc5 = nn.Linear(20, 20)
        self.fc6 = nn.Linear(20, 20)
        self.fc7 = nn.Linear(20, 20)
        self.fc8 = nn.Linear(20, 20)
        self.fc9 = nn.Linear(20, 1)

    def forward(self, x):
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        x = torch.tanh(self.fc3(x))
        x = torch.tanh(self.fc4(x))
        x = torch.tanh(self.fc5(x))
        x = torch.tanh(self.fc6(x))
        x = torch.tanh(self.fc7(x))
        x = torch.tanh(self.fc8(x))
        x = self.fc9(x)
        return x


def lossfunction(u_ICBC_NN, u_ICBC, u_F, ut_F, ux_F, uxx_F):
    """
    Parameters
    ----------
    u_ICBC_NN : torch.Tensor of size 100x1
        The prediction for u of the neural network for all training points at the initial and boundary position
    u_ICBC : torch.Tensor of size 100x1
        The true values for u on the 100 training points at the initial and boundary position
    u_F : torch.Tensor of size 10000x1
        The prediction for u of the neural network for all collocation points.
    ut_F : torch.Tensor of size 10000x1
        The first derivative of u_F w.r.t. t.
    ux_F : torch.Tensor of size 10000x1
        The first derivative of u_F w.r.t. x.
    uxx_F : torch.Tensor of size 10000x1
        The second derivative of u_F w.r.t. x.
    Returns
    -------
    loss = MSE_F + MSE_ICBC.

    """
    MSE_ICBC = torch.sum((u_ICBC_NN - u_ICBC) * (u_ICBC_NN - u_ICBC)) / u_ICBC_NN.shape[0]
    F = (ut_F + u_F * ux_F - (0.01 / 3.14159265359) * uxx_F) * (
            ut_F + u_F * ux_F - (0.01 / 3.14159265359) * uxx_F)
    MSE_F = torch.sum(F) / F.shape[0]
    # print("MSE_F: %.7f, MSE_ICBC: %.7f" %(MSE_F, MSE_ICBC)) # <- uncomment for debugging
    return MSE_ICBC + MSE_F


def train_model(path="./burgers_shock.mat"):
    # setting a seed for pytorch as well as one for numpy
    torch.manual_seed(2)
    np.random.seed(2)

    # hyperparameters
    POINTS = 1500
    NUM_EPOCHS = 2000
    LEARNING_RATE = 0.002
    # number of datapoints from initial and boundary conditions
    ICBC_POINTS = 100
    # and number of collocation points (used for MSE_F)
    F_POINTS = 10000

    # loading the data
    x, t, x_mesh, t_mesh, u_target_mesh, Phi, u_target, \
    Phi_ICBC, u_ICBC, Phi_F = prep_data(path, ICBC_POINTS, F_POINTS)

    # transforming the needed arrays into torch.Tensors
    Phi = torch.from_numpy(Phi).float()
    Phi_ICBC = torch.from_numpy(Phi_ICBC).float()
    u_ICBC = torch.from_numpy(u_ICBC).float()
    Phi_F = torch.from_numpy(Phi_F).float()

    # plotting the solution for u to get a feeling of how the data looks like
    plt.figure()
    plt.contourf(t_mesh, x_mesh, u_target_mesh)
    plt.title("True Solution")
    plt.xlabel("t")
    plt.ylabel("x")
    plt.tight_layout()

    # tracking all operations on Phi_col so that we can backpropagate through the network
    Phi_F.requires_grad = True

    # creating an instance of our neural network class
    model = Net()

    # creating a list for predictions and an array for our loss function values
    predictions = []
    losses = np.zeros(NUM_EPOCHS)

    # creating an instance of the Adam optimizer with the specified learning rate
    optimizer = Adam(model.parameters(), lr=LEARNING_RATE)
    # The scheduler will reduce the learning rate of the optimizer by a factor of 0.5 after 1000 epochs
    scheduler = StepLR(optimizer, step_size=1000, gamma=0.5)

    for epoch in range(1, NUM_EPOCHS + 1):
        # making a prediction for u at the collocation points
        u_F_NN = model(Phi_F)

        # computing the first derivatives of u_F_NN w.r.t. x and t by backpropagating through the network
        g = grad(u_F_NN.sum(), Phi_F, create_graph=True)[0]

        # This computes the first derivatives of du_col_pred/dx w.r.t. x and t.
        # The first column of gg corresponds to the second derivative w.r.t. x
        gg = grad(g[:, 0:1], Phi_F, grad_outputs=torch.ones(u_F_NN.shape), create_graph=True)[0]

        # These are our derivatives of u_F_NN w.r.t. x, t and the second derivative w.r.t. x
        ux_F = g[:, 0:1]
        ut_F = g[:, 1:2]
        uxx_F = gg[:, 0:1]

        optimizer.zero_grad()

        # making a prediction for u at the initial & boundary points
        u_ICBC_NN = model(Phi_ICBC)

        # computing the loss
        loss = lossfunction(u_ICBC_NN, u_ICBC, u_F_NN, ut_F, ux_F, uxx_F)

        # propagating backward
        loss.backward()

        # updating parameters
        optimizer.step()

        # updating learning rate
        scheduler.step()
        losses[epoch - 1] = loss.detach().numpy()

        # Save predicted fields every 100 epochs for plotting later
        if epoch % 100 == 0:
            print("Epoch: %d, Loss: %.7f" % (epoch, losses[epoch - 1]))
            with torch.no_grad():
                u_pred = model(Phi)
                u_pred = u_pred.detach().numpy()
                predictions.append([u_pred.reshape(u_target_mesh.shape), epoch])

    torch.save(model.state_dict(), "my_Burgers_model.pt")
    # plot the predictions we saved for every 100 epochs
    # f, ((ax0, ax1, ax2, ax3, ax4), (ax5, ax6, ax7, ax8, ax9)) = plt.subplots(2, 5)
    # for i in range(0, 20, 2):
    #     ime = "ax" + str(int(i / 2))
    #     if i == 0:
    #         vars()[ime].contourf(t_mesh, x_mesh, u_target_mesh)
    #         vars()[ime].set_title('original data')
    #     else:
    #         vars()[ime].contourf(t_mesh, x_mesh, predictions[i][0])
    #         vars()[ime].set_title('epoch ' + str(predictions[i][1]))
    #     # vars()[ime].contourf(t_mesh,x_mesh,predictions[i][0])
    #     vars()[ime].set_xlabel('t')
    #     vars()[ime].set_ylabel('x')
    #     # vars()[ime].set_title('epoch ' + str(predictions[i][1]))

    # plot the loss value over the epochs
    plt.tight_layout()
    plt.figure()
    plt.plot(losses)
    plt.title("Training Loss")
    plt.xlabel('Epochs')
    plt.ylabel('Loss')

    # plot the predictions u of the last epoch over different time steps specified in timesteps
    u_pred = u_pred.reshape((100, 256))
    f, ((ax0, ax1), (ax2, ax3)) = plt.subplots(2, 2)
    timesteps = np.array([0, 33, 66, 99])

    ax0.plot(x, u_target_mesh[timesteps[0], :])
    ax0.plot(x, u_pred[timesteps[0], :])
    ax0.set_title('t = ' + str(t[0][0]))
    ax0.set_xlabel('x')
    ax0.set_ylabel('u')
    ax0.legend(('u_target', 'NN prediction'))

    ax1.plot(x, u_target_mesh[timesteps[1], :])
    ax1.plot(x, u_pred[timesteps[1], :])
    ax1.set_title('t = ' + str(t[33][0]))
    ax1.set_xlabel('x')
    ax1.set_ylabel('u')
    ax1.legend(('u_target', 'NN prediction'))

    ax2.plot(x, u_target_mesh[timesteps[2], :])
    ax2.plot(x, u_pred[timesteps[2], :])
    ax2.set_title('t = ' + str(t[66][0]))
    ax2.set_xlabel('x')
    ax2.set_ylabel('u')
    ax2.legend(('u_target', 'NN prediction'))

    ax3.plot(x, u_target_mesh[timesteps[3], :])
    ax3.plot(x, u_pred[timesteps[3], :])
    ax3.set_title('t = ' + str(t[99][0]))
    ax3.set_xlabel('x')
    ax3.set_ylabel('u')
    ax3.legend(('u_target', 'NN prediction'))
    plt.tight_layout()
    plt.show()

    # creating validation grid
    # x_grid = np.arange(-1, 1, 1 / 500)
    # t_grid = np.arange(0, 1, 1 / 500)
    # x_mesh, t_mesh = np.meshgrid(x_grid, t_grid)
    # Phi_val = np.concatenate([x_mesh.reshape(-1, 1), t_mesh.reshape(-1, 1)], axis=1)
    # Phi_val = torch.from_numpy(Phi_val).float()
    # u_val_pred = model(Phi_val)



def main():
    train_model()


if __name__ == "__main__":
    main()
