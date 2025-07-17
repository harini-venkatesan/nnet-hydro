import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pickle
from matplotlib.animation import FuncAnimation, PillowWriter

with open('pkl-files/loglik_grid_parallel-30.pkl', 'rb') as f:
    loglik_grid = pickle.load(f)


with open('pkl-files/loglik_grid_parallel-120.pkl', 'rb') as f:
    loglik_grid2 = pickle.load(f)

param_ranges = [np.linspace(-3, 3, 100) for _ in range(3)]

X, Y = np.meshgrid(param_ranges[0], param_ranges[1], indexing='ij')

likelihood_grid1 = np.exp(loglik_grid)
likelihood_grid2 = np.exp(loglik_grid2)

# nnet_grid = np.exp(nnet_grid)
global_min = np.nanmin(likelihood_grid1)
global_max = np.nanmax(likelihood_grid1)

print(likelihood_grid1.shape)
fig = plt.figure(figsize=(8, 6))
# ax = fig.add_subplot(111, projection='3d')
ax1 = fig.add_subplot(121, projection='3d')
ax2 = fig.add_subplot(122, projection='3d')

# def update_surface_1(k):
    # ax1.clear()

# frames = np.linspace(0, 99, 20, dtype=int)
# anim = FuncAnimation(fig, update_surface, frames=frames, interval=800, blit=False)

with open('pkl-files/nnet-grid.pkl', 'rb') as f:
    nnet_grid1 = pickle.load(f)

print(nnet_grid1.shape)
nnet_grid1 = np.exp(nnet_grid1)

#to flip?
# nnet_grid1 = nnet_grid1[:, :, ::-1]

global_min1 = np.nanmin(nnet_grid1)
global_max1 = np.nanmax(nnet_grid1)

def update_surface(k):
    # Clear both axes to avoid overplotting
    ax1.clear()
    ax2.clear()
    
    # Difference surface on ax1
    Z_diff = likelihood_grid1[:, :, k]
    surf1 = ax1.plot_surface(X, Y, Z_diff, cmap='viridis', edgecolor='none')
    ax1.set_xlabel('parameter 1')
    ax1.set_ylabel('parameter 2')
    ax1.set_zlabel('lh')
    ax1.set_title(f'evaluated lh at param 3 = {param_ranges[2][k]:.2f}')
    ax1.set_zlim(global_min, global_max)
    
    # nnet surface on ax2
    Z_nnet = nnet_grid1[:, :, k]
    surf2 = ax2.plot_surface(X, Y, Z_nnet, cmap='viridis', edgecolor='none')
    ax2.set_xlabel('parameter 1')
    ax2.set_ylabel('parameter 2')
    ax2.set_zlabel('nnet output')
    ax2.set_title(f'shrek nnet at param 3 = {param_ranges[2][k]:.2f}')
    ax2.set_zlim(global_min1, global_max1)
    
    # Return both for FuncAnimation blitting compatibility
    return surf1, surf2

frames = np.linspace(0, 99, 20, dtype=int)
anim = FuncAnimation(fig, update_surface, frames=frames, interval=800, blit=False)


#save as gif
# anim.save('trained-mcmc.gif', writer=PillowWriter(fps=2))

#save as mp4
# anim.save('trained-single.mp4', writer='ffmpeg', fps=2)

plt.show()


# import numpy as np
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
# import pickle

# # Load your saved grids
# with open('nnet-grid-1.pkl', 'rb') as f:
#     loglik_grid = pickle.load(f)  # raw NN outputs, probably log-likelihood

# with open('nnet-grid-2.pkl', 'rb') as f:
#     nnet_grid1 = pickle.load(f)  # another grid to compare to

# # Make sure shapes match or handle accordingly
# assert loglik_grid.shape == nnet_grid1.shape, "Grid shapes must match!"

# # If your nnet_grid1 needs flipping to align axes (check carefully)
# # nnet_grid1_aligned = nnet_grid1[:, :, ::-1]

# # Option 1: Work in log scale (no exponentiation)
# # Z1 = loglik_grid
# # Z2 = nnet_grid1_aligned

# # Option 2: Work in likelihood scale (exponentiate both)
# Z1 = np.exp(loglik_grid)
# Z2 = np.exp(nnet_grid1)

# # Optional: Normalize each slice per frame for visualization
# def normalize(Z):
#     Zmin, Zmax = np.nanmin(Z), np.nanmax(Z)
#     return (Z - Zmin) / (Zmax - Zmin + 1e-8)

# # Create param grids for X, Y
# param_ranges = [np.linspace(-3, 3, 100) for _ in range(3)]
# X, Y = np.meshgrid(param_ranges[0], param_ranges[1], indexing='ij')

# import matplotlib.animation as animation

# fig = plt.figure(figsize=(12, 6))
# ax1 = fig.add_subplot(121, projection='3d')
# ax2 = fig.add_subplot(122, projection='3d')

# # Compute global min/max over all slices for consistent zlim
# global_min = min(np.nanmin(Z1), np.nanmin(Z2))
# global_max = max(np.nanmax(Z1), np.nanmax(Z2))

# # global_min1 = np.nanmin(nnet_grid1)
# # global_max1 = np.nanmax(nnet_grid1)


# def update_surface(k):
#     ax1.clear()
#     ax2.clear()

#     # Slice along the 3rd parameter axis
#     Z1_slice = Z1[:, :, k]
#     Z2_slice = Z2[:, :, k]

#     # Normalize for plotting (optional)
#     # Z1_slice = normalize(Z1_slice)
#     # Z2_slice = normalize(Z2_slice)

#     surf1 = ax1.plot_surface(X, Y, Z1_slice, cmap='viridis', edgecolor='none')
#     ax1.set_title(f'Grid 1 at param3 = {param_ranges[2][k]:.2f}')
#     ax1.set_xlabel('Parameter 1')
#     ax1.set_ylabel('Parameter 2')
#     ax1.set_zlabel('Value')
#     ax1.set_zlim(global_min, global_max)

#     surf2 = ax2.plot_surface(X, Y, Z2_slice, cmap='viridis', edgecolor='none')
#     ax2.set_title(f'Grid 2 at param3 = {param_ranges[2][k]:.2f}')
#     ax2.set_xlabel('Parameter 1')
#     ax2.set_ylabel('Parameter 2')
#     ax2.set_zlabel('Value')
#     ax2.set_zlim(global_min, global_max)

#     return surf1, surf2

# frames = np.linspace(0, 99, 20, dtype=int)
# anim = animation.FuncAnimation(fig, update_surface, frames=frames, interval=800, blit=False)

# plt.show()
