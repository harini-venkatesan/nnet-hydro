import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
import pickle
from matplotlib.animation import FuncAnimation, PillowWriter

# # Load the log-likelihood grid
# with open('loglik_grid_parallel.pkl', 'rb') as f:
#     loglik_grid = pickle.load(f)

# # Set up your parameter ranges (should match computation)
# param_ranges = [np.linspace(-3, 3, 100) for _ in range(3)]

# X, Y = np.meshgrid(param_ranges[0], param_ranges[1], indexing='ij')

# fig = plt.figure(figsize=(8, 6))
# ax = fig.add_subplot(111, projection='3d')

# def update_surface(k):
#     ax.clear()
#     Z = np.exp(loglik_grid[:, :, k])  # Take exp to get likelihood
#     surf = ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none')
#     ax.set_xlabel('Parameter 1')
#     ax.set_ylabel('Parameter 2')
#     ax.set_zlabel('Likelihood')
#     ax.set_title(f'Likelihood Surface at Param 3 = {param_ranges[2][k]:.2f}')
#     ax.set_zlim(np.nanmin(Z), np.nanmax(Z))
#     return surf,


# Load the log-likelihood grid
with open('loglik_grid_parallel.pkl', 'rb') as f:
    loglik_grid = pickle.load(f)

# Set up your parameter ranges (should match computation)
param_ranges = [np.linspace(-3, 3, 100) for _ in range(3)]

X, Y = np.meshgrid(param_ranges[0], param_ranges[1], indexing='ij')

# Precompute the likelihood grid and global min/max for Z
likelihood_grid = np.exp(loglik_grid)
global_min = np.nanmin(likelihood_grid)
global_max = np.nanmax(likelihood_grid)

fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')

def update_surface(k):
    ax.clear()
    Z = likelihood_grid[:, :, k]
    surf = ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none')
    ax.set_xlabel('Parameter 1')
    ax.set_ylabel('Parameter 2')
    ax.set_zlabel('Likelihood')
    ax.set_title(f'Likelihood Surface at Param 3 = {param_ranges[2][k]:.2f}')
    ax.set_zlim(global_min, global_max)  # Fix z-axis range for all frames
    return surf,
# anim = FuncAnimation(fig, update_surface, frames=20, interval=800, blit=False)

frames = np.linspace(0, 99, 20, dtype=int)  # 20 evenly spaced indices
anim = FuncAnimation(fig, update_surface, frames=frames, interval=800, blit=False)

# To display in a notebook, use:
# from IPython.display import HTML
# HTML(anim.to_jshtml())

# To save as GIF
# anim.save('loglik_surface.gif', writer=PillowWriter(fps=2))

# To save as MP4 (requires ffmpeg)
# anim.save('loglik_surface.mp4', writer='ffmpeg', fps=2)

plt.show()