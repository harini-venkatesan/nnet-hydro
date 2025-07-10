import os
os.environ["MKL_NUM_THREADS"] = "1" 
os.environ["NUMEXPR_NUM_THREADS"] = "1" 
os.environ["OMP_NUM_THREADS"] = "1" 

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
from shrek import *
import numpy as np
import multiprocessing as mp
from tqdm import tqdm

tt = 0
def my_loglik(my_model, theta, datapoints, data, sigma):
    """
    This returns the log-likelihood of my_model given theta,
    datapoints, the observed data and sigma. It uses the
    model_wrapper function to do a model solve.
    """
    # Ensure theta is a numpy array
    if isinstance(theta, torch.Tensor):
        theta_np = theta.detach().cpu().numpy()
    else:
        theta_np = np.asarray(theta)
    output = model_wrapper(my_model, theta_np, datapoints)
    # Ensure output and data are numpy arrays
    output = np.asarray(output)
    data = np.asarray(data)
    ret = - (0.5 / sigma ** 2) * np.sum((output - data) ** 2)
    return ret


#goes from coarsest to finest
resolutions = [(120,120)]
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

llh = genllhfn(my_models[0],datapoints,data,sigma)

# def llh(params):
#     # Replace this stub with your actual likelihood computation
#     # params is a 1D array or list of length 3
#     return -np.sum((np.array(params) - np.array([1, 2, 3]))**2)

def worker(params_chunk):
    return [llh(params) for params in params_chunk]

if __name__ == "__main__":
    # # Example: generate 10,000 samples, each with 3 parameters
    n_samples = 10000
    # n_params = 3
    # samples = np.random.randn(n_samples, n_params)
    with open('june-samples/mcmc-long-1', 'rb') as f:
        samples = pickle.load(f)

    samples = np.array(samples)
    samples = samples[0]
    
    # Split samples for parallel processing
    n_cpu = mp.cpu_count()
    chunk_size = int(np.ceil(n_samples / n_cpu))
    chunks = [samples[i*chunk_size:(i+1)*chunk_size] for i in range(n_cpu)]

    with mp.Pool(n_cpu) as pool:
        results = list(tqdm(pool.imap(worker, chunks), total=len(chunks)))
    
    # Flatten and save results
    llh_values = np.concatenate(results)
    np.savetxt("llh_values.txt", llh_values)
    print(f"Saved {len(llh_values)} log-likelihood values to llh_values.txt")