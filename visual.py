import numpy as np
import matplotlib.pyplot as plt
import pickle

# Load the data
with open('samples/long-test-1', 'rb') as f:
    data = pickle.load(f)
data = np.array(data)
# Reshape to (5, 10, 10000, 3) as per the code in ess.py
data = data.reshape(5, 10, data.shape[1], 3)
# Slice to get the first run (index 0, 0)
run_data = data[0, 0, :, :]  # shape: (10000, 3)
# Extract the first two parameters
param1 = run_data[:, 0]
param2 = run_data[:, 1]
# Plot scatter of the two parameters
plt.scatter(param1, param2, alpha=0.5)
plt.xlabel('Parameter 1')
plt.ylabel('Parameter 2')
plt.title('Scatter plot of first two parameters for the first run')
plt.show()

# Compute mean and std for each parameter
mean_params = run_data.mean(axis=0)
std_params = run_data.std(axis=0)
print(f"Mean of parameters: {mean_params}")
print(f"Std of parameters: {std_params}")

# Plot histograms for each parameter
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for i in range(3):
    axes[i].hist(run_data[:, i], bins=30, alpha=0.7)
    axes[i].set_title(f'Parameter {i+1}')
    axes[i].set_xlabel(f'Param {i+1}')
    axes[i].set_ylabel('Frequency')
plt.tight_layout()
plt.show()

# Overlay the mean on the scatter plot
plt.scatter(param1, param2, alpha=0.5, label='Samples')
plt.scatter(mean_params[0], mean_params[1], color='red', label='Mean', s=100, marker='x')
plt.xlabel('Parameter 1')
plt.ylabel('Parameter 2')
plt.title('Scatter plot of first two parameters for the first run')
plt.legend()
plt.show()

# Compute mean for each parameter across all 100 runs
all_means = data.mean(axis=2)  # shape: (5, 10, 3)
# Reshape to (50, 3) for easier handling
all_means_reshaped = all_means.reshape(-1, 3)
# Compute the mean of means and the standard error of the mean
mean_of_means = all_means_reshaped.mean(axis=0)
std_of_means = all_means_reshaped.std(axis=0)
std_error = std_of_means / np.sqrt(all_means_reshaped.shape[0])
print(f"Mean of means for each parameter: {mean_of_means}")
print(f"Standard error of the mean for each parameter: {std_error}")






