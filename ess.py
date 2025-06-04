import numpy as np
import matplotlib.pyplot as plt
import pickle

import pickle
import numpy as np
import arviz as az
import os
import glob
import xarray as xr
import pickle
import sys
from scipy.stats import sem
cd = os.getcwd()
import matplotlib.pyplot as plt
from scipy import stats

def get_ess(data,n_draws):
    n_samples=n_draws*10
    n_points=20
    first_draw = data.draw.values[0]
    xdata = np.linspace(n_samples / n_points, n_samples, n_points)
    draw_divisions = np.linspace(n_draws // n_points, n_draws, n_points, dtype=int)

    ess_dataset = xr.concat(
                [
                    az.ess(
                        data.sel(draw=slice(first_draw + draw_div)),
                        method="bulk",
                    )
                    for draw_div in draw_divisions
                ],
                dim="ess_dim",
            )

    ess_tail_dataset = xr.concat(
                [
                    az.ess(
                        data.sel(draw=slice(first_draw + draw_div)),
                        method="tail",
                    )
                    for draw_div in draw_divisions
                ],
                dim="ess_dim",
            )

    return ess_dataset,ess_tail_dataset
    
    
with open('samples/long-test-1', 'rb') as f:
    data = pickle.load(f)
    
data = np.array(data)
data = data.reshape(5,10,data.shape[1],3)
data = data[:,:,:8000,:]
print(data.shape)
ess_1 = []
ess_2 = []
ess_3 = []

tail_1 = []
tail_2 = []
tail_3 = []

'''
samps = az.convert_to_dataset(np.array(data), group = 'posterior')
ess,tail = get_ess(samps, data.shape[1])

print(ess, tail)


x = np.linspace(0, 1000, 20)  # 20 points from 0 to 1000
param_names = ['param1', 'param2']

fig, axs = plt.subplots(2, 2, figsize=(12, 6), sharex=True)

# Bulk ESS
axs[0, 0].plot(x, ess['x'][:, 0], label='Bulk ESS', marker='o')
axs[0, 0].set_title(f'Bulk ESS - {param_names[0]}')
axs[0, 0].set_ylabel('ESS')

axs[0, 1].plot(x, ess['x'][:, 1], label='Bulk ESS', marker='o')
axs[0, 1].set_title(f'Bulk ESS - {param_names[1]}')

# Tail ESS
axs[1, 0].plot(x, tail['x'][:, 0], label='Tail ESS', color='orange', marker='s')
axs[1, 0].set_title(f'Tail ESS - {param_names[0]}')
axs[1, 0].set_xlabel('X')
axs[1, 0].set_ylabel('ESS')

axs[1, 1].plot(x, tail['x'][:, 1], label='Tail ESS', color='orange', marker='s')
axs[1, 1].set_title(f'Tail ESS - {param_names[1]}')
axs[1, 1].set_xlabel('X')

plt.tight_layout()


plt.savefig('test-1.png')

'''
for i in data: 
    samps = az.convert_to_dataset(np.array(i), group = 'posterior')
    ess,tail = get_ess(samps, data.shape[2])
    ess_1.append(ess['x'][:,0])
    ess_2.append(ess['x'][:,1])
    ess_3.append(ess['x'][:,2])
    tail_1.append(tail['x'][:,0])
    tail_2.append(tail['x'][:,1])
    tail_3.append(tail['x'][:,2])


ess_1 = np.array(ess_1)
ess_2 = np.array(ess_2)
ess_3 = np.array(ess_3)

tail_1 = np.array(tail_1)
tail_2 = np.array(tail_2)
tail_3 = np.array(tail_3)


mean_ess1 = np.mean(ess_1, axis=0)
mean_ess2 = np.mean(ess_2, axis=0)
mean_ess3 = np.mean(ess_3, axis=0)
mean_tail1 = np.mean(tail_1, axis=0)
mean_tail2 = np.mean(tail_2, axis=0)
mean_tail3 = np.mean(tail_3, axis=0)

print(mean_ess1[-1]/700)
print(mean_ess2[-1]/700)
print(mean_ess3[-1]/700)
print(mean_tail1[-1]/700)
print(mean_tail2[-1]/700)
print(mean_tail3[-1]/700)

# Compute SEM
sem_ess1 = stats.sem(ess_1, axis=0)
sem_ess2 = stats.sem(ess_2, axis=0)
sem_ess3 = stats.sem(ess_3, axis=0)
sem_tail1 = stats.sem(tail_1, axis=0)
sem_tail2 = stats.sem(tail_2, axis=0)
sem_tail3 = stats.sem(tail_3, axis=0)

# Plotting

x = np.linspace(0, 700, 20)  # 20 points from 0 to 1000
fig, axs = plt.subplots(2, 3, figsize=(12, 6), sharex=True)

axs[0, 0].errorbar(x, mean_ess1, yerr=sem_ess1, fmt='-o', label='ESS 1')
axs[0, 0].set_title('Bulk ESS - param1')
axs[0, 0].set_ylabel('ESS')

axs[0, 1].errorbar(x, mean_ess2, yerr=sem_ess2, fmt='-o', label='ESS 2')
axs[0, 1].set_title('Bulk ESS - param2')

axs[0, 2].errorbar(x, mean_ess3, yerr=sem_ess3, fmt='-o', label='ESS 3')
axs[0, 2].set_title('Bulk ESS - param3')


axs[1, 0].errorbar(x, mean_tail1, yerr=sem_tail1, fmt='-s', color='orange', label='Tail ESS 1')
axs[1, 0].set_title('Tail ESS - param1')
axs[1, 0].set_ylabel('ESS')
axs[1, 0].set_xlabel('time')

axs[1, 1].errorbar(x, mean_tail2, yerr=sem_tail2, fmt='-s', color='orange', label='Tail ESS 2')
axs[1, 1].set_title('Tail ESS - param2')
axs[1, 1].set_xlabel('time')

axs[1, 2].errorbar(x, mean_tail3, yerr=sem_tail3, fmt='-s', color='orange', label='Tail ESS 3')
axs[1, 2].set_title('Tail ESS - param3')
axs[1, 2].set_xlabel('time')


plt.tight_layout()
# plt.show()
'''
print(ess_1, ess_2, tail_1, tail_2)


param_names = ['param1', 'param2']

fig, axs = plt.subplots(2, 2, figsize=(12, 6), sharex=True)

# Bulk ESS
axs[0, 0].plot(x, ess[:, 0], label='Bulk ESS', marker='o')
axs[0, 0].set_title(f'Bulk ESS - {param_names[0]}')
axs[0, 0].set_ylabel('ESS')

axs[0, 1].plot(x, ess[:, 1], label='Bulk ESS', marker='o')
axs[0, 1].set_title(f'Bulk ESS - {param_names[1]}')

# Tail ESS
axs[1, 0].plot(x, tail[:, 0], label='Tail ESS', color='orange', marker='s')
axs[1, 0].set_title(f'Tail ESS - {param_names[0]}')
axs[1, 0].set_xlabel('X')
axs[1, 0].set_ylabel('ESS')

axs[1, 1].plot(x, tail[:, 1], label='Tail ESS', color='orange', marker='s')
axs[1, 1].set_title(f'Tail ESS - {param_names[1]}')
axs[1, 1].set_xlabel('X')

plt.tight_layout()
'''


plt.savefig('single-3.png')

