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
        # Create a single neural network and share it across levels
        self.base_net = SimpleNet()
        self.nnet = [self.base_net for _ in range(self.J)] 
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
        
        # Adaptive proposal parameters
        self.adaptation_interval = 20
        self.warmup_steps = 100  # Number of steps before starting adaptation
        self.max_adaptation_samples = 1000  # Maximum number of samples to keep for adaptation
        self.samples_for_adaptation = []
        self.adaptation_counter = 0
        self.adaptation_rate = 1.0  # Initial adaptation rate
        self.min_adaptation_rate = 0.01  # Minimum adaptation rate
        
    def update_proposal_covariance(self):
        """Update the proposal covariance based on collected samples from innermost level"""
        if len(self.samples_for_adaptation) < 2:
            return
            
        # Convert samples to numpy array
        samples_array = torch.stack(self.samples_for_adaptation).detach().cpu().numpy()
        
        # Calculate empirical covariance
        emp_cov = np.cov(samples_array.T)
        
        # Add small jitter to ensure positive definiteness
        jitter = 1e-6 * np.eye(emp_cov.shape[0])
        emp_cov = emp_cov + jitter
        
        # Scale the covariance (optional tuning parameter)
        scale = 2.4**2 / emp_cov.shape[0]  # Optimal scaling for multivariate normal
        emp_cov = emp_cov * scale
        
        # Update the proposal covariance with adaptation rate
        new_cov = torch.tensor(emp_cov, dtype=torch.float32)
        self.proposal_covariance = (1 - self.adaptation_rate) * self.proposal_covariance + self.adaptation_rate * new_cov
        
        # Reduce adaptation rate
        self.adaptation_rate = max(self.min_adaptation_rate, self.adaptation_rate * 0.995)
        
        # Clear the samples for next adaptation
        self.samples_for_adaptation = []
        
    def Loss(self, current, proposed): 
        return (proposed - current)
    
    def proposal(self, oldx):
        # oldx: torch tensor
        min_val = torch.tensor([-3,-3,-3], dtype=oldx.dtype, device=oldx.device) 
        max_val = torch.tensor([3,3,3], dtype=oldx.dtype, device=oldx.device)
        
        # Use the current proposal covariance
        m = MultivariateNormal(oldx, self.proposal_covariance)
        newx = m.sample()
        
        if torch.all(newx > min_val) and torch.all(newx < max_val):
            return newx
        else:
            return oldx
        
    def _llh(self, llh_fn, x):
        # Converts torch tensor to numpy for model likelihood
        if isinstance(x, torch.Tensor):
            x_np = x.detach().cpu().numpy()
        else:
            x_np = np.asarray(x)
        return llh_fn(x_np)

    def update_neural_networks(self, current_j, current_sample, proposed_sample, loss):
        """Update neural networks using the loss from recursive sampling"""
        if current_j == self.J:
            return
            
        # Use the loss from recursive sampling
        if self.reject_flag[current_j]:
            loss.backward()
            self.optimizer[current_j].step()
            self.optimizer[current_j].zero_grad()

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
        
        for step in range(self.N):
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
            
            self.update_neural_networks(current_j, current_sample, proposed_sample, loss)
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
                
                # Store samples for adaptation at innermost level
                if self.total_num > self.warmup_steps:
                    self.samples_for_adaptation.append(current_sample.clone())
                    if len(self.samples_for_adaptation) > self.max_adaptation_samples:
                        self.samples_for_adaptation.pop(0)  # Remove oldest sample
                    
                    self.adaptation_counter += 1
                    
                    # Update proposal covariance every adaptation_interval steps
                    if self.adaptation_counter >= self.adaptation_interval:
                        self.update_proposal_covariance()
                        self.adaptation_counter = 0
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
                self.update_neural_networks(current_j, current_sample, proposed_sample, loss)
            self.lnrho[current_j-1].append(curr_omega)

        return current_sample, currllh, firstinner, inner_loss
    
    def update(self, loss, optimizer, reject_flag):
        loss.backward()
        if reject_flag:
            optimizer.step()
            optimizer.zero_grad()
            
