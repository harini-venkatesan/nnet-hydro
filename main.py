import os
os.environ["MKL_NUM_THREADS"] = "4" 
os.environ["NUMEXPR_NUM_THREADS"] = "4" 
os.environ["OMP_NUM_THREADS"] = "4" 

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
resolutions = [(10,10),(30,30),(120,120)]
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

proposal_covariance = torch.tensor([0.01, 0.01, 0.01], dtype=torch.float32)
prop_cov = torch.diag(proposal_covariance)


def multiple_runs(chain_id): 
    np.random.seed(chain_id)
    torch.manual_seed(chain_id)
    
    # Reset random seed for consistent true_parameters
    torch.manual_seed(data_seed)
    my_models[-1].solve()
    true_parameters = my_models[-1].random_process.parameters

    x0 = torch.tensor(true_parameters, dtype=torch.float32)    
    llh_levels = [level3, level2, level1]
    N = 10000
    M = 5
    J = 1

    nt1 = time.time()
    sampler = ShrekMCMC(x0, llh_levels, N, M, J, prop_cov)
    sampler.shrek()
    nt2 = time.time()
    print('time per run', nt2-nt1)

    print('acceptance rates', sampler.acceptance_rate)

    samples_outer = np.array([s.detach().numpy() for s in sampler.samples[0]])
    print(samples_outer.shape)
    return samples_outer, sampler

if __name__ == '__main__':

    nchains = 50

    # with Pool(4) as p: 
    #     samples_test = p.starmap(multiple_runs, zip(range(nchains)))

    samples_test, sampler = multiple_runs(12)
        
    with open('july-samples/replay-buffer-2', 'wb') as f: 
        pickle.dump(samples_test, f) 

    param_ranges = [np.linspace(-3, 3, 100) for _ in range(3)]

    def worker_loglik(args):
        theta, i, j, k = args
        theta_tensor = torch.tensor(theta, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            val = sampler.nnet[0](theta_tensor).item()
        return (i, j, k, val)

    grid_args = []
    for i, x in enumerate(param_ranges[0]):
        for j, y in enumerate(param_ranges[1]):
            for k, z in enumerate(param_ranges[2]):
                theta = np.array([x, y, z])
                grid_args.append((theta, i, j, k))

    loglik_grid = np.zeros((100, 100, 100))

    with Pool(processes=4) as pool:
        results = pool.map(worker_loglik, grid_args)

    for i, j, k, val in results:
        loglik_grid[i, j, k] = val

    with open('replay-1.pkl', 'wb') as f:
        pickle.dump(loglik_grid, f)     


    # t1 = time.time()
    # #with Pool(1) as p: 
    # nestedsamps = schemata(range(1),0)
    # t2 = time.time()
    # print(nestedsamps)
    # print('time for mcmc is: ', t2-t1)
    


# import os
# os.environ["MKL_NUM_THREADS"] = "1" 
# os.environ["NUMEXPR_NUM_THREADS"] = "1" 
# os.environ["OMP_NUM_THREADS"] = "1" 

# import time
# from itertools import product
# import matplotlib.pyplot as plt
# import numpy as np
# import json
# from model import *
# import torch
# import pickle
# from model import Model, model_wrapper, project_eigenpairs
# from multiprocessing import Pool
# from itertools import repeat
# from shrek import *

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


# #goes from coarsest to finest
# resolutions = [(10,10), (30,30), (120,120)]
# # resolutions = [(10,10)]
# # Set random field parameters
# field_mean = 0
# field_stdev = 1
# lamb_cov = 0.05

# # Set the number of unknown parameters (i.e. dimension of theta in posterior)
# nparam = 3

# sigma = 0.01

# # Data generation seed
# data_seed = 123446
# points_list = [0.1, 0.3, 0.5, 0.7, 0.9]

# # Note this can take several minutes for large resolutions
# my_models = []
# for r in resolutions:
#     my_models.append(Model(r, field_mean, field_stdev, nparam, lamb_cov))

# # Project eignevactors from fine model to all coarse models
# for i in range(len(my_models[:-1])):
#     project_eigenpairs(my_models[-1], my_models[i])


# # Solve finest model as a test and plot transmissivity field and solution
# # torch.random.seed(data_seed)
# torch.manual_seed(data_seed)
# np.random.seed(data_seed)
# my_models[-1].solve()

# #calculate for 120x120 resolution
# true_parameters = my_models[-1].random_process.parameters
# print("true params", true_parameters)
# # Define the sampling points.

# x_data = y_data = torch.tensor(points_list)
# datapoints = torch.tensor(list(product(x_data, y_data)))

# # Get data from the sampling points and perturb it with some noise.
# # noise = torch.random.normal(0, 0.001, len(datapoints))
# noise = 0.001 * torch.randn(len(datapoints))
# noise = noise.detach().cpu().numpy()
# # Generate data from the finest model for use in pymc3 inference - these data are used in all levels
# data = model_wrapper(my_models[-1], true_parameters, datapoints) + noise
# # result = model_wrapper(my_models[-1], true_parameters, datapoints)
# # print(type(result), type(noise))
# # print(np.shape(result), np.shape(noise))

# def genllhfn(model,datapoints,data,sigma):
#     def _llh(x):
#         return my_loglik(model, x, datapoints, data, sigma)
#     return _llh

# level3 = genllhfn(my_models[2],datapoints,data,sigma)
# level2 = genllhfn(my_models[1],datapoints,data,sigma)
# level1 = genllhfn(my_models[0],datapoints,data,sigma)

# proposal_covariance = torch.tensor([0.01, 0.01, 0.01], dtype=torch.float32)
# prop_cov = torch.diag(proposal_covariance)

# # def multiple_runs(chain_id, x0): 
 
# # if __name__ == '__main__':
    
# x0 = torch.tensor(true_parameters, dtype=torch.float32)
# print(x0)

# # nchains = 10
# print("n = 10000, M = 5, J = 1")

# # samples_test = multiple_runs(data_seed, x0)
    
# # np.random.seed(chain_id)
# # torch.manual_seed(chain_id)    

# llh_levels = [level3, level2, level1]
# N = 5000
# M = 5
# J = 1
# print('running mcmc for samples: ',N)

# nt1 = time.time()
# sampler = ShrekMCMC(x0, llh_levels, N, M, J, prop_cov)
# sampler.shrek()
# nt2 = time.time()
# print('time per run', nt2-nt1)

# print('acceptance rates', sampler.acceptance_rate)
# samples_test = sampler.samples
# samples_outer = np.array([s.detach().numpy() for s in sampler.samples[0]])
# print(samples_outer.shape)
#     # return samples_outer

# with open('samples/long-single-2', 'wb') as f: 
#     pickle.dump(samples_test, f) 

# # t1 = time.time()
# # #with Pool(1) as p: 
# # nestedsamps = schemata(range(1),0)
# # t2 = time.time()
# # print(nestedsamps)
# # print('time for mcmc is: ', t2-t1)

# # param_ranges = [np.linspace(-3, 3, 100) for _ in range(3)]

# # def worker_loglik(args):
# #     theta, i, j, k = args
# #     theta_tensor = torch.tensor(theta, dtype=torch.float32).unsqueeze(0)
# #     with torch.no_grad():
# #         val = sampler.nnet[0](theta_tensor).item()
# #     return (i, j, k, val)



# # grid_args = []
# # for i, x in enumerate(param_ranges[0]):
# #     for j, y in enumerate(param_ranges[1]):
# #         for k, z in enumerate(param_ranges[2]):
# #             theta = np.array([x, y, z])
# #             grid_args.append((theta, i, j, k))

# # loglik_grid = np.zeros((100, 100, 100))

# # with Pool(processes=10) as pool:
# #     results = pool.map(worker_loglik, grid_args)

# # for i, j, k, val in results:
# #     loglik_grid[i, j, k] = val

# # with open('nnet-grid-1.pkl', 'wb') as f:
# #     pickle.dump(loglik_grid, f)     


# '''
# import os
# os.environ["MKL_NUM_THREADS"] = "1"
# os.environ["NUMEXPR_NUM_THREADS"] = "1"
# os.environ["OMP_NUM_THREADS"] = "1"

# import time
# from itertools import product
# import matplotlib.pyplot as plt
# import numpy as np
# import json
# from model import *
# import torch
# import pickle
# from multiprocessing import Pool, cpu_count

# tt = 0
# def my_loglik(my_model, theta, datapoints, data, sigma):
#     if isinstance(theta, torch.Tensor):
#         theta_np = theta.detach().cpu().numpy()
#     else:
#         theta_np = np.asarray(theta)
#     output = model_wrapper(my_model, theta_np, datapoints)
#     output = np.asarray(output)
#     data = np.asarray(data)
#     ret = - (0.5 / sigma ** 2) * np.sum((output - data) ** 2)
#     return ret

# resolutions = [(120,120)]
# field_mean = 0
# field_stdev = 1
# lamb_cov = 0.05
# nparam = 3
# sigma = 0.01
# data_seed = 123446
# points_list = [0.1, 0.3, 0.5, 0.7, 0.9]

# my_models = []
# for r in resolutions:
#     my_models.append(Model(r, field_mean, field_stdev, nparam, lamb_cov))

# for i in range(len(my_models[:-1])):
#     project_eigenpairs(my_models[-1], my_models[i])

# torch.manual_seed(data_seed)
# np.random.seed(data_seed)
# my_models[-1].solve()

# true_parameters = my_models[-1].random_process.parameters
# print("true params", true_parameters)
# x_data = y_data = torch.tensor(points_list)
# datapoints = torch.tensor(list(product(x_data, y_data)))
# noise = 0.001 * torch.randn(len(datapoints))
# noise = noise.detach().cpu().numpy()
# data = model_wrapper(my_models[-1], true_parameters, datapoints) + noise

# def genllhfn(model,datapoints,data,sigma):
#     def _llh(x):
#         return my_loglik(model, x, datapoints, data, sigma)
#     return _llh

# level1 = genllhfn(my_models[0],datapoints,data,sigma)
# llh_fn = level1

# param_ranges = [np.linspace(-3, 3, 300) for _ in range(3)]

# def worker_loglik(args):
#     theta, i, j, k = args
#     val = llh_fn(theta)
#     return (i, j, k, val)

# if __name__ == "__main__":
#     grid_args = []
#     for i, x in enumerate(param_ranges[0]):
#         for j, y in enumerate(param_ranges[1]):
#             for k, z in enumerate(param_ranges[2]):
#                 theta = np.array([x, y, z])
#                 grid_args.append((theta, i, j, k))

#     loglik_grid = np.zeros((300, 300, 300))

#     with Pool(processes=24) as pool:
#         results = pool.map(worker_loglik, grid_args)

#     for i, j, k, val in results:
#         loglik_grid[i, j, k] = val

#     with open('loglik_grid_parallel-long-120-2.pkl', 'wb') as f:
#         pickle.dump(loglik_grid, f)     
# '''