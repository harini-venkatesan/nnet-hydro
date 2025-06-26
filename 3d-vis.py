import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pickle
from matplotlib.animation import FuncAnimation, PillowWriter

# with open('loglik_grid_parallel-120.pkl', 'rb') as f:
#     loglik_grid = pickle.load(f)

# param_ranges = [np.linspace(-3, 3, 100) for _ in range(3)]

# X, Y = np.meshgrid(param_ranges[0], param_ranges[1], indexing='ij')

# fig = plt.figure(figsize=(8, 6))
# ax = fig.add_subplot(111, projection='3d')

# def update_surface(k):
#     ax.clear()
#     Z = np.exp(loglik_grid[:, :, k]) 
#     surf = ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none')
#     ax.set_xlabel('parameter 1')
#     ax.set_ylabel('parameter 2')
#     ax.set_zlabel('llh')
#     ax.set_title(f'llh surface at param 3 = {param_ranges[2][k]:.2f}')
#     ax.set_zlim(np.nanmin(Z), np.nanmax(Z))
#     return surf


with open('loglik_grid_parallel-30.pkl', 'rb') as f:
    nnet_grid = pickle.load(f)

with open('loglik_grid_parallel-120.pkl', 'rb') as f:
    loglik_grid = pickle.load(f)

param_ranges = [np.linspace(-3, 3, 100) for _ in range(3)]

X, Y = np.meshgrid(param_ranges[0], param_ranges[1], indexing='ij')

likelihood_grid = np.exp(loglik_grid)
nnet_grid = np.exp(nnet_grid)
global_min = np.nanmin(likelihood_grid)
global_max = np.nanmax(likelihood_grid)

fig = plt.figure(figsize=(8, 6))
# ax = fig.add_subplot(111, projection='3d')
ax1 = fig.add_subplot(121, projection='3d')
ax2 = fig.add_subplot(122, projection='3d')

# def update_surface_1(k):
    # ax1.clear()

# frames = np.linspace(0, 99, 20, dtype=int)
# anim = FuncAnimation(fig, update_surface, frames=frames, interval=800, blit=False)

with open('nnet-grid-2.pkl', 'rb') as f:
    nnet_grid1 = pickle.load(f)

param_ranges = [np.linspace(-3, 3, 100) for _ in range(3)]

X, Y = np.meshgrid(param_ranges[0], param_ranges[1], indexing='ij')

nnet_grid1 = np.exp(nnet_grid1)
# nnet_grid = np.exp(nnet_grid)
# global_min = np.nanmin(likelihood_grid)
# global_max = np.nanmax(likelihood_grid)

# fig = plt.figure(figsize=(8, 6))
# ax = fig.add_subplot(111, projection='3d')

# def update_surface(k):
#     # ax2.clear()
#     Z = likelihood_grid[:, :, k] - nnet_grid[:,:,k]
#     surf = ax1.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none')
#     ax1.set_xlabel('parameter 1')
#     ax1.set_ylabel('parameter 2')
#     ax1.set_zlabel('llh')
#     ax1.set_title(f'llh surface at param 3 = {param_ranges[2][k]:.2f}')
#     ax1.set_zlim(global_min, global_max)  

#     Z = nnet_grid[:,:,k]
#     surf = ax2.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none')
#     ax2.set_xlabel('parameter 1')
#     ax2.set_ylabel('parameter 2')
#     ax2.set_zlabel('llh')
#     ax2.set_title(f'llh surface at param 3 = {param_ranges[2][k]:.2f}')
#     ax2.set_zlim(global_min, global_max)  
#     return surf

def update_surface(k):
    # Clear both axes to avoid overplotting
    ax1.clear()
    ax2.clear()
    
    # Difference surface on ax1
    Z_diff = likelihood_grid[:, :, k] - nnet_grid[:, :, k]
    surf1 = ax1.plot_surface(X, Y, Z_diff, cmap='viridis', edgecolor='none')
    ax1.set_xlabel('parameter 1')
    ax1.set_ylabel('parameter 2')
    ax1.set_zlabel('llh diff')
    ax1.set_title(f'Differenceat param 3 = {param_ranges[2][k]:.2f}')
    ax1.set_zlim(global_min, global_max)
    
    # nnet surface on ax2
    Z_nnet = nnet_grid1[:, :, k]
    surf2 = ax2.plot_surface(X, Y, Z_nnet, cmap='viridis', edgecolor='none')
    ax2.set_xlabel('parameter 1')
    ax2.set_ylabel('parameter 2')
    ax2.set_zlabel('llh')
    ax2.set_title(f'nnet at param 3 = {param_ranges[2][k]:.2f}')
    ax2.set_zlim(global_min, global_max)
    
    # Return both for FuncAnimation blitting compatibility
    return surf1, surf2

frames = np.linspace(0, 99, 20, dtype=int)
anim = FuncAnimation(fig, update_surface, frames=frames, interval=800, blit=False)


#save as gif
# anim.save('loglik_surface.gif', writer=PillowWriter(fps=2))

#save as mp4
# anim.save('loglik_surface.mp4', writer='ffmpeg', fps=2)

plt.show()