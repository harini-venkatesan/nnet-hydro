import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions.multivariate_normal import MultivariateNormal
from collections import deque
import random

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
            nn.Linear(124, 1)
        )
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
        
        self.base_net = SimpleNet()
        self.nnet = [self.base_net for _ in range(self.J)] 
        self.optimizer = [optim.Adam(self.nnet[i].parameters(), lr=1e-4) for i in range(self.J)]
        
        self.llh_levels = likelihood_levels
        self.proposal_covariance = proposal_covariance
        self.sample_covariance = proposal_covariance
        
        self.samples = [[] for _ in range(self.J+1)] 
        self.lnrho = [[] for _ in range(self.J)] 
        self.total_num = 0 
        
        self.x0 = x0.clone().detach() if isinstance(x0, torch.Tensor) else torch.tensor(x0, dtype=torch.float32)
        
        self.v = []
        self.s = []
        self.running_sum = torch.zeros_like(self.x0)
        self.acceptance_rate = []
        self.rejected_samples = []
        self.reject_flag = [False for _ in range(self.J+1)]

        # Adaptive proposal
        self.adaptation_interval = 20
        self.warmup_steps = 100
        self.max_adaptation_samples = 1000
        self.samples_for_adaptation = []
        self.adaptation_counter = 0
        self.adaptation_rate = 1.0
        self.min_adaptation_rate = 0.01

        # Batch loss accumulation
        self.loss_batch_size = 10
        self.loss_accumulator = [0.0 for _ in range(self.J)]
        self.step_counter = [0 for _ in range(self.J)]

        # Tracking for diagnostics
        self.running_loss = [[] for _ in range(self.J)]
        self.grad_norms = [[] for _ in range(self.J)]

    def update_proposal_covariance(self):
        if len(self.samples_for_adaptation) < 2:
            return
        samples_array = torch.stack(self.samples_for_adaptation).detach().cpu().numpy()
        emp_cov = np.cov(samples_array.T) + 1e-6 * np.eye(samples_array.shape[1])
        scale = 2.4**2 / emp_cov.shape[0]
        emp_cov *= scale
        new_cov = torch.tensor(emp_cov, dtype=torch.float32)
        self.proposal_covariance = (1 - self.adaptation_rate) * self.proposal_covariance + self.adaptation_rate * new_cov
        self.adaptation_rate = max(self.min_adaptation_rate, self.adaptation_rate * 0.995)
        self.samples_for_adaptation = []

    def Loss(self, current, proposed):
        return proposed - current

    def proposal(self, oldx):
        min_val = torch.tensor([-3, -3, -3], dtype=oldx.dtype, device=oldx.device) 
        max_val = torch.tensor([3, 3, 3], dtype=oldx.dtype, device=oldx.device)
        m = MultivariateNormal(oldx, self.proposal_covariance)
        newx = m.sample()
        return newx if torch.all(newx > min_val) and torch.all(newx < max_val) else oldx

    def _llh(self, llh_fn, x):
        x_np = x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else np.asarray(x)
        return llh_fn(x_np)

    def update_neural_networks(self, current_j, loss):
        if current_j == self.J:
            return

        self.loss_accumulator[current_j] += loss
        self.step_counter[current_j] += 1

        if self.step_counter[current_j] >= self.loss_batch_size:
            total_loss = self.loss_accumulator[current_j] / self.loss_batch_size
            total_loss.backward()

            # Clip gradients and track norm
            total_norm = 0.0
            for param in self.nnet[current_j].parameters():
                if param.grad is not None:
                    param_norm = param.grad.data.norm(2).item()
                    total_norm += param_norm**2
            total_norm = total_norm**0.5
            self.grad_norms[current_j].append(total_norm)

            torch.nn.utils.clip_grad_norm_(self.nnet[current_j].parameters(), max_norm=10.0)
            self.optimizer[current_j].step()
            self.optimizer[current_j].zero_grad()

            self.running_loss[current_j].append(total_loss.item())
            self.loss_accumulator[current_j] = 0.0
            self.step_counter[current_j] = 0

    def shrek(self):
        current_j = 0
        current_sample = self.x0.clone()

        for i in range(self.J):
            self.lnrho[i].append(self.nnet[i](current_sample))
            self.s.append(0)
            self.v.append(0)
            self.acceptance_rate.append(0)
        self.acceptance_rate.append(0)

        oldllh = torch.tensor(self._llh(self.llh_levels[current_j], current_sample), dtype=torch.float32)

        for step in range(self.N):
            proposed_sample, newinner, oldinner, loss = self.shrek_recursive(current_j+1, current_sample)
            newllh = torch.tensor(self._llh(self.llh_levels[current_j], proposed_sample), dtype=torch.float32)

            logflip = (newllh - oldllh) + (oldinner - newinner)

            if torch.log(torch.rand(1)) < logflip:
                current_sample = proposed_sample.clone()
                oldllh = newllh
                self.acceptance_rate[current_j] += 1
                self.reject_flag[current_j] = False
            else:
                self.rejected_samples.append(proposed_sample.clone())
                self.reject_flag[current_j] = True

            self.update_neural_networks(current_j, loss)
            self.samples[current_j].append(current_sample.clone())

        # Flush remaining loss
        for j in range(self.J):
            if self.step_counter[j] > 0:
                total_loss = self.loss_accumulator[j] / self.step_counter[j]
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.nnet[j].parameters(), max_norm=10.0)
                self.optimizer[j].step()
                self.optimizer[j].zero_grad()
                self.running_loss[j].append(total_loss.item())
                self.grad_norms[j].append(total_loss.grad.norm().item() if total_loss.grad is not None else 0.0)
                self.loss_accumulator[j] = 0.0
                self.step_counter[j] = 0

    def shrek_recursive(self, current_j, current_sample):
        curr_omega = self.nnet[current_j - 1](current_sample)
        firstinner_val = self._llh(self.llh_levels[current_j], current_sample)
        firstinner = torch.logaddexp(torch.tensor(firstinner_val, dtype=torch.float32), curr_omega)
        currllh = firstinner

        for i in range(self.M):
            if current_j == self.J:
                self.total_num += 1
                proposed_sample = self.proposal(current_sample)

                if self.total_num > self.warmup_steps:
                    self.samples_for_adaptation.append(current_sample.clone())
                    if len(self.samples_for_adaptation) > self.max_adaptation_samples:
                        self.samples_for_adaptation.pop(0)
                    self.adaptation_counter += 1
                    if self.adaptation_counter >= self.adaptation_interval:
                        self.update_proposal_covariance()
                        self.adaptation_counter = 0
            else:
                proposed_sample, newinner, oldinner, loss = self.shrek_recursive(current_j + 1, current_sample)

            prop_omega = self.nnet[current_j - 1](proposed_sample)
            newllh_val = self._llh(self.llh_levels[current_j], proposed_sample)
            newllh = torch.logaddexp(torch.tensor(newllh_val, dtype=torch.float32), prop_omega)

            logflip = (newllh - currllh) if current_j == self.J else (newllh - currllh) + (oldinner - newinner)

            if torch.log(torch.rand(1)) < logflip:
                self.acceptance_rate[current_j] += 1
                current_sample = proposed_sample.clone()
                currllh = newllh
                self.reject_flag[current_j] = False
            else:
                self.reject_flag[current_j] = True

            self.samples[current_j].append(current_sample.clone())
            # inner_loss = self.Loss(firstinner, currllh)

            inner_loss = self.Loss(firstinner, currllh)
            inner_loss = inner_loss.clone().detach().requires_grad_(True)

            if current_j != self.J:
                self.update_neural_networks(current_j, loss)

            # if current_j != self.J:
            #     self.replay_buffer[current_j - 1].append((current_sample.detach(), inner_loss.detach()))
            #     self.update_neural_networks_from_replay(current_j - 1)

            self.lnrho[current_j - 1].append(curr_omega)

        return current_sample, currllh, firstinner, inner_loss
