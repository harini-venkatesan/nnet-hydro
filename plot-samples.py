import matplotlib.pyplot as plt
import numpy as np
import pickle 

with open('../june-samples/batch-single-2', 'rb') as f:
    samples = pickle.load(f)

samples = np.array(samples)
print(samples.shape)
# samples = np.transpose(samples)
# samples = np.transpose(samples, (0, 2, 1))
# samples = samples[1]
# print(samples.shape)  
mean_values = np.mean(samples, axis=0)

# Calculate mean for each parameter across all runs and samples
# param_means = np.mean(samples, axis=(0, 1))  

print("Mean for each parameter:", mean_values)
# 3D Scatter plot

fig = plt.figure(figsize=(7,6))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(samples[:, 0], samples[:, 1], samples[:, 2], s=4, alpha=0.6)
ax.set_xlabel('param 1')
ax.set_ylabel('param 2')
ax.set_zlabel('param 3')
ax.set_title('points')
plt.tight_layout()
# plt.savefig('mcmc-samps.png')
plt.show()