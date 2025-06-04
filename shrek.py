import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.autograd import Variable
import time
from itertools import repeat
from multiprocessing import Pool

class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(3, 124),
            nn.ReLU(),
            nn.Linear(124, 124),
            nn.ReLU(),
            nn.Linear(124, 124),
            nn.ReLU(),
            nn.Linear(124, 1))
        # nn.Softplus())        
        self._init_weights()

    def _init_weights(self):
        for layer in self.layers:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)


    def forward(self, x):
        return self.layers(x)
            
            
class ShrekMCMC: 
    def __init__(self, x0, likelihood_levels, N, M, J, proposal_covariance):
        self.M = M
        self.N = N
        self.J = J
        self.nnet = [SimpleNet() for _ in range(self.J)] 
        self.optimizer = [optim.Adam(self.nnet[i].parameters(), lr = 1e-4) for i in range(self.J)]
        self.llh_levels = likelihood_levels
        self.proposal_covariance = proposal_covariance
        self.sample_covariance = proposal_covariance
        self.samples = [[] for _ in range(self.J+1)] 
        self.lnrho = [[] for _ in range(self.J)] 
        self.total_num = 0 
        # Ensure initial sample is torch tensor for the net, but store numpy for the model
        if isinstance(x0, torch.Tensor):
            self.x0 = x0.clone().detach()
        else:
            self.x0 = torch.tensor(x0, dtype=torch.float32)
        self.v = []
        self.s = []
        self.running_sum = torch.zeros_like(self.x0)
        self.acceptance_rate = []
        self.rejected_samples = []
        self.reject_flag =  [False for _ in range(self.J+1)] 
        
    def Loss(self, current, proposed): 
        return (proposed - current)
    
    def proposal(self, oldx):
        # oldx: torch tensor
        min_val = torch.tensor([-3,-3,-3], dtype=oldx.dtype, device=oldx.device) 
        max_val = torch.tensor([3,3,3], dtype=oldx.dtype, device=oldx.device)
        m = MultivariateNormal(oldx, self.proposal_covariance)
        newx = m.sample()
        if torch.all(newx > min_val) and torch.all(newx < max_val):
            return newx
        else:
            return oldx
        
    def adaptive_prop(self, current_sample):
        # ... [unchanged] ...
        pass  # You can use as is

    def _llh(self, llh_fn, x):
        # Converts torch tensor to numpy for model likelihood
        if isinstance(x, torch.Tensor):
            x_np = x.detach().cpu().numpy()
        else:
            x_np = np.asarray(x)
        return llh_fn(x_np)

    def shrek(self):
        current_j = 0
        current_sample = self.x0.clone()
        
        #initialize with current values 
        for i in range(self.J):
            # Neural net expects torch tensor
            self.lnrho[i].append(self.nnet[i](current_sample))
            self.s.append(0)
            self.v.append(0)
            self.acceptance_rate.append(0)

        self.acceptance_rate.append(0)
        
        oldllh = self._llh(self.llh_levels[current_j], current_sample)
        oldllh = torch.tensor(oldllh, dtype=torch.float32)  # for PyTorch math below
        
        for _ in range(self.N):
            proposed_sample, newinner, oldinner, loss = self.shrek_recursive(current_j+1, current_sample)
            newllh = self._llh(self.llh_levels[current_j], proposed_sample)
            newllh = torch.tensor(newllh, dtype=torch.float32)

            logflip = (newllh - oldllh) + (oldinner - newinner)

            if torch.log(torch.rand(1)) < logflip:
                current_sample = proposed_sample.clone()
                oldllh = newllh
                self.acceptance_rate[current_j]+=1
                self.reject_flag[current_j] = False
            else: 
                self.rejected_samples.append(proposed_sample.clone())
                self.reject_flag[current_j] = True
            
            self.update(loss, self.optimizer[current_j], self.reject_flag[current_j])
            self.samples[current_j].append(current_sample.clone()) 

    def shrek_recursive(self, current_j, current_sample):
        # For neural network: always use torch tensor
        curr_omega = self.nnet[current_j-1](current_sample)

        # For model likelihood: always use numpy
        firstinner_val = self._llh(self.llh_levels[current_j], current_sample)
        firstinner = torch.logaddexp(torch.tensor(firstinner_val, dtype=torch.float32), curr_omega)
        currllh = firstinner
        
        for i in range(self.M):
            if current_j == self.J:  # innermost proposal
                self.total_num += 1
                proposed_sample = self.proposal(current_sample)
            else:
                proposed_sample, newinner, oldinner, loss = self.shrek_recursive(current_j+1, current_sample)
            
            prop_omega = self.nnet[current_j-1](proposed_sample)
            newllh_val = self._llh(self.llh_levels[current_j], proposed_sample)
            newllh = torch.logaddexp(torch.tensor(newllh_val, dtype=torch.float32), prop_omega)

            if current_j == self.J:
                logflip = newllh - currllh
            else: 
                logflip = (newllh - currllh) + (oldinner - newinner)

            if torch.log(torch.rand(1)) < logflip:
                self.acceptance_rate[current_j] += 1
                current_sample = proposed_sample.clone()
                currllh = newllh
                self.reject_flag[current_j] = False
            else:
                self.reject_flag[current_j] = True

            self.samples[current_j].append(current_sample.clone())

            inner_loss = self.Loss(firstinner, currllh)
            if current_j != self.J:
                self.update(loss, self.optimizer[current_j], self.reject_flag[current_j])
            self.lnrho[current_j-1].append(curr_omega)

        return current_sample, currllh, firstinner, inner_loss
    
    def update(self, loss, optimizer, reject_flag):
        loss.backward()
        if reject_flag:
            optimizer.step()
            optimizer.zero_grad()
            
            
# class ShrekMCMC: 
#     def __init__(self, x0, likelihood_levels, N, M, J, proposal_covariance):
#         self.M = M
#         self.N = N
#         self.J = J
#         self.nnet = [SimpleNet() for _ in range(self.J)] 
#         self.optimizer = [optim.Adam(self.nnet[i].parameters(), lr = 1e-4) for i in range(self.J)]
#         self.llh_levels = likelihood_levels
#         self.proposal_covariance = proposal_covariance
#         self.sample_covariance = proposal_covariance
#         # self.samples = []
#         self.samples = [[] for _ in range(self.J+1)] 
#         self.lnrho = [[] for _ in range(self.J)] 
#         self.total_num = 0 
#         self.x0 = x0
#         self.v = []
#         self.s = []
#         self.running_sum = torch.zeros_like(self.x0)
#         self.acceptance_rate = []
#         self.rejected_samples = []
#         self.reject_flag =  [False for _ in range(self.J+1)] 
        
#     def Loss(self, current, proposed): 
#         return (proposed - current)
    
#     def proposal(self, oldx):
#         min_val = torch.tensor([-3,-3,-3]) 
#         max_val = torch.tensor([3,3,3])
#         #range_size = (max_val - min_val)  # 2
#         #np.random.rand(N) * range_size + min_val
#         # newx = np.random.multivariate_normal(oldx, self.proposal_covariance)
#         m = MultivariateNormal(oldx, self.proposal_covariance)
#         newx = m.sample()
#         if torch.all(newx > min_val) and torch.all(newx < max_val):
#             return newx
#         else:
#             return oldx
        
#     def adaptive_prop(self, current_sample):
#         if self.total_num != 0:
#             delta = current_sample - self.running_sum
#             self.running_sum += delta / self.total_num
#             outer_product = torch.outer(delta, delta)
#             self.sample_covariance += (outer_product - self.sample_covariance) / self.total_num
#         if self.total_num == 50:
#             self.proposal_covariance = (2.4**2/2)*(self.sample_covariance + torch.diag(torch.ones(len(current_sample))) * 1e-6)
#             self.total_num = 0
#         return 

#     def shrek(self):
#         current_j = 0
#         current_sample = self.x0.clone()
        
#         #initialize with current values 
#         for i in range(self.J):
#             self.lnrho[i].append(self.nnet[i](current_sample))
#             self.s.append(0)
#             self.v.append(0)
#             self.acceptance_rate.append(0)

#         self.acceptance_rate.append(0)
        
#         oldllh = self.llh_levels[current_j](current_sample)
        
#         for _ in range(self.N):
#             proposed_sample, newinner, oldinner, loss = self.shrek_recursive(current_j+1,current_sample)
#             newllh = self.llh_levels[current_j](proposed_sample)
            
#             logflip = (newllh - oldllh) + (oldinner - newinner)

#             if torch.log(torch.rand(1)) < logflip:
#                 current_sample = proposed_sample
#                 oldllh = newllh
#                 self.acceptance_rate[current_j]+=1
#                 self.reject_flag[current_j] = False

#             else: 
#                 self.rejected_samples.append(proposed_sample)
#                 self.reject_flag[current_j] = True
            
#             self.update(loss, self.optimizer[current_j], self.reject_flag[current_j])
#             # self.loss_list.append(loss.item())
            
#             self.samples[current_j].append(current_sample) 


#     def shrek_recursive(self,current_j,current_sample):
#         #with torch.no_grad():
#         curr_omega = self.nnet[current_j-1](current_sample)

#         firstinner = torch.logaddexp(self.llh_levels[current_j](current_sample),curr_omega)
#         currllh = firstinner
        
#         for i in range(self.M):
#             if current_j == self.J: #inner most proposal -- BASE CASE
#                 self.total_num+=1
#                 proposed_sample = self.proposal(current_sample)

#             else:
#                 proposed_sample,newinner,oldinner,loss = self.shrek_recursive(current_j+1,current_sample)
            
#             #with torch.no_grad():
#             prop_omega = self.nnet[current_j-1](proposed_sample)

#             newllh = torch.logaddexp(self.llh_levels[current_j](proposed_sample),prop_omega)
            
#             if current_j == self.J: logflip = newllh - currllh
#             else: 
#                 logflip = (newllh - currllh) + (oldinner - newinner)

#             if torch.log(torch.rand(1)) < logflip:
#                 self.acceptance_rate[current_j]+=1
#                 current_sample = proposed_sample 
#                 # newinner = newllh
#                 currllh = newllh
#                 self.reject_flag[current_j] = False
#             else: 
#                 self.reject_flag[current_j] = True
                
#             self.samples[current_j].append(current_sample)
            
#             # if current_j == self.J: self.adaptive_prop(current_sample)
#             #update lnrho
#             inner_loss = self.Loss(firstinner, currllh)
#             if current_j != self.J:
#                 self.update(loss, self.optimizer[current_j], self.reject_flag[current_j])
                
#             self.lnrho[current_j-1].append(curr_omega)

#         return current_sample,currllh,firstinner,inner_loss
    
#     def update(self, loss, optimizer, reject_flag):
#         loss.backward()
            
#         if reject_flag == True:
#             optimizer.step()
#             # for param in self.nnet.parameters():
#             #     param.grad = None

#             optimizer.zero_grad()