import matplotlib.pyplot as plt
import numpy as np
import torch
from shrek import *
from model import *

import time
from itertools import product
import matplotlib.pyplot as plt
import numpy as np
import json
from model import *
import torch
import pickle
from model import Model, model_wrapper, project_eigenpairs
from multiprocessing import Pool
from itertools import repeat

import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
from matplotlib.animation import PillowWriter

def eval_nnet_on_grid(shrek_obj, net_idx=0, grid_points=100, device='cpu'):
    """
    Evaluate a single neural net on a 3D grid in [-3,3]^3.
    Returns numpy array of shape (grid_points, grid_points, grid_points)
    """
    net = shrek_obj.nnet[net_idx]
    net.eval()
    net.to(device)

    vals = torch.linspace(-3, 3, grid_points, device=device)
    X, Y, Z = torch.meshgrid(vals, vals, vals, indexing='ij')

    # Flatten the grid for batch evaluation
    inputs = torch.stack([X.reshape(-1), Y.reshape(-1), Z.reshape(-1)], dim=1)

    with torch.no_grad():
        outputs = net(inputs).squeeze()
    outputs_np = outputs.cpu().numpy().reshape(grid_points, grid_points, grid_points)
    return outputs_np, vals.cpu().numpy()

def plot_3d_slice_animation(llh_grid, nnet_grid, param_vals, frames=20, save_path=None):
    """
    Animate slice of 3D surfaces along the 3rd dimension (Z-axis)
    llh_grid, nnet_grid: 3D numpy arrays shape=(N,N,N)
    param_vals: 1D array of length N for the third dimension values
    """

    X, Y = np.meshgrid(param_vals, param_vals, indexing='ij')
    global_min = np.nanmin(llh_grid)
    global_max = np.nanmax(llh_grid)
    global_min_nnet = np.nanmin(nnet_grid)
    global_max_nnet = np.nanmax(nnet_grid)

    fig = plt.figure(figsize=(12, 6))
    ax1 = fig.add_subplot(121, projection='3d')
    ax2 = fig.add_subplot(122, projection='3d')

    def update(k):
        ax1.clear()
        ax2.clear()

        # Slice at fixed Z = param_vals[k]
        Z_slice_llh = np.exp(llh_grid[:, :, k])
        Z_slice_nnet = np.exp(nnet_grid[:, :, k])

        surf1 = ax1.plot_surface(X, Y, Z_slice_llh, cmap='viridis', edgecolor='none')
        ax1.set_xlabel('param 1')
        ax1.set_ylabel('param 2')
        ax1.set_zlabel('likelihood')
        ax1.set_title(f'True likelihood at param 3 = {param_vals[k]:.2f}')
        ax1.set_zlim(global_min, global_max)

        surf2 = ax2.plot_surface(X, Y, Z_slice_nnet, cmap='viridis', edgecolor='none')
        ax2.set_xlabel('param 1')
        ax2.set_ylabel('param 2')
        ax2.set_zlabel('nn output')
        ax2.set_title(f'NN prediction at param 3 = {param_vals[k]:.2f}')
        ax2.set_zlim(global_min_nnet, global_max_nnet)

        return surf1, surf2

    anim = FuncAnimation(fig, update, frames=np.linspace(0, llh_grid.shape[2]-1, frames, dtype=int), interval=800)
    anim.save('trained-mcmc.gif', writer=PillowWriter(fps=2))

    # if save_path:
    #     anim.save(save_path, writer='ffmpeg', fps=2)
    
    # plt.show()



# def plot_nnet_output(shrek_obj, j=0, fixed_dim=2, fixed_val=0.0, grid_size=100):
#     # j = which neural net in the list
#     # fixed_dim = which input dimension to fix (0,1, or 2)
#     # fixed_val = value to fix that dimension at
#     # grid_size = number of points along each axis

#     net = shrek_obj.nnet[j]
#     net.eval()

#     # Prepare grid for the other two dims
#     dims = [0, 1, 2]
#     dims.remove(fixed_dim)

#     vals1 = torch.linspace(-3, 3, grid_size)
#     vals2 = torch.linspace(-3, 3, grid_size)

#     grid1, grid2 = torch.meshgrid(vals1, vals2, indexing="ij")
#     inputs = torch.zeros(grid_size * grid_size, 3)

#     inputs[:, fixed_dim] = fixed_val
#     inputs[:, dims[0]] = grid1.reshape(-1)
#     inputs[:, dims[1]] = grid2.reshape(-1)

#     with torch.no_grad():
#         outputs = net(inputs).squeeze()

#     outputs_grid = outputs.reshape(grid_size, grid_size).numpy()

#     plt.figure(figsize=(8, 6))
#     plt.contourf(vals1.numpy(), vals2.numpy(), outputs_grid, levels=50, cmap='viridis')
#     plt.colorbar()
#     plt.xlabel(f'Input dim {dims[0]}')
#     plt.ylabel(f'Input dim {dims[1]}')
#     plt.title(f'Neural net output (net {j}) fixing dim {fixed_dim}={fixed_val}')
#     plt.savefig('test.png')

def load_all_nnets(shrek_obj, filename="all_nnets.pkl"):
    with open(filename, "rb") as f:
        nets_state = pickle.load(f)
    
    for net, state in zip(shrek_obj.nnet, nets_state):
        net.load_state_dict(state)


# tt = 0
# def my_loglik(my_model, theta, datapoints, data, sigma):
#     """
#     This returns the log-likelihood of my_model given theta,
#     datapoints, the observed data and sigma. It uses the
#     model_wrapper function to do a model solve.
#     """
#     # Ensure theta is a numpy array
#     if isinstance(theta, torch.Tensor):
#         theta_np = theta.detach().cpu().numpy()
#     else:
#         theta_np = np.asarray(theta)
#     output = model_wrapper(my_model, theta_np, datapoints)
#     # Ensure output and data are numpy arrays
#     output = np.asarray(output)
#     data = np.asarray(data)
#     ret = - (0.5 / sigma ** 2) * np.sum((output - data) ** 2)
#     return ret


#goes from coarsest to finest
resolutions = [(10,10),(30,30),(20,20)]
# resolutions = [(10,10)]
# Set random field parameters
field_mean = 0
field_stdev = 1
lamb_cov = 0.05

# Set the number of unknown parameters (i.e. dimension of theta in posterior)
nparam = 3

sigma = 0.01

# Data generation seed
data_seed = 123446
points_list = [0.1, 0.3, 0.5, 0.7, 0.9]

# Note this can take several minutes for large resolutions
my_models = []
for r in resolutions:
    my_models.append(Model(r, field_mean, field_stdev, nparam, lamb_cov))

# Project eignevactors from fine model to all coarse models
for i in range(len(my_models[:-1])):
    project_eigenpairs(my_models[-1], my_models[i])


# Solve finest model as a test and plot transmissivity field and solution
# torch.random.seed(data_seed)
torch.manual_seed(data_seed)
np.random.seed(data_seed)
my_models[-1].solve()

#calculate for 120x120 resolution
true_parameters = my_models[-1].random_process.parameters
print(true_parameters)
# Define the sampling points.

x_data = y_data = torch.tensor(points_list)
datapoints = torch.tensor(list(product(x_data, y_data)))

# Get data from the sampling points and perturb it with some noise.
# noise = torch.random.normal(0, 0.001, len(datapoints))
noise = 0.001 * torch.randn(len(datapoints))
noise = noise.detach().cpu().numpy()
# Generate data from the finest model for use in pymc3 inference - these data are used in all levels
data = model_wrapper(my_models[-1], true_parameters, datapoints) + noise
# result = model_wrapper(my_models[-1], true_parameters, datapoints)
# print(type(result), type(noise))
# print(np.shape(result), np.shape(noise))

def genllhfn(model,datapoints,data,sigma):
    # Returns a function that takes a parameter vector (numpy or torch) and outputs a scalar (float)
    def _llh(x):
        return my_loglik(model, x, datapoints, data, sigma)
    return _llh

level3 = genllhfn(my_models[2],datapoints,data,sigma)
level2 = genllhfn(my_models[1],datapoints,data,sigma)
level1 = genllhfn(my_models[0],datapoints,data,sigma)
x0 = torch.tensor(true_parameters, dtype=torch.float32)    
llh_levels = [level3, level2, level1]
J = 1
N = 10
M = 5

proposal_covariance = torch.tensor([0.01, 0.01, 0.01], dtype=torch.float32)
prop_cov = torch.diag(proposal_covariance)

shrek_loaded = ShrekMCMC(x0, llh_levels, N, M, J, proposal_covariance, batch_size=100)
load_all_nnets(shrek_loaded, "all_nnets.pkl")
# plot_nnet_output(shrek_loaded)

nnet_grid, param_vals = eval_nnet_on_grid(shrek_loaded, net_idx=0, grid_points=100)

with open('loglik_grid_parallel-120.pkl', 'rb') as f:
    loglik_grid = pickle.load(f)

param_ranges = [np.linspace(-3, 3, 100) for _ in range(3)]

X, Y = np.meshgrid(param_ranges[0], param_ranges[1], indexing='ij')

likelihood_grid = np.exp(loglik_grid)
global_min = np.nanmin(likelihood_grid)
global_max = np.nanmax(likelihood_grid)

# true_llh_grid = nnet_grid + 0.1 * np.random.randn(*nnet_grid.shape)  # just demo
plot_3d_slice_animation(likelihood_grid, nnet_grid, param_vals, frames=20)
