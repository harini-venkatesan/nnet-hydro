import matplotlib.pyplot as plt
import numpy as np
import pickle 

with open('june-samples/AEM-single', 'rb') as f:
    samples = pickle.load(f)

samples = np.array(samples)
print(samples.shape)

samples = np.transpose(samples, (0, 2, 1))
samples = samples[10]
print(samples.shape)  # For sanity check

# 3D Scatter plot
fig = plt.figure(figsize=(7,6))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(samples[:, 0], samples[:, 1], samples[:, 2], s=4, alpha=0.6)
ax.set_xlabel('Parameter 1')
ax.set_ylabel('Parameter 2')
ax.set_zlabel('Parameter 3')
ax.set_title('Sampled Points in Parameter Space')
plt.tight_layout()
plt.show()