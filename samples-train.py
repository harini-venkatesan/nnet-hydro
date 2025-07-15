import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import pickle
import matplotlib.animation as animation

# features: shape (50, 10000, 3)
# targets: shape (50, 10000, 1)
num_runs = 100
num_samples = 10000
num_params = 3

with open('june-samples/mcmc-long-test-1', 'rb') as f:
    samples = pickle.load(f)

with open('june-samples/llh_values-long-1.pkl', 'rb') as f:
    llh_vals = pickle.load(f)

# Example parameters: fill these with your actual data
# features = np.random.uniform(-3, 3, (num_samples, num_params)).astype(np.float32)
features = np.array(samples)
# features = np.transpose(features)
# features = features[0]
# targets = np.exp(-np.sum(features**2, axis=-1, keepdims=True))  # just an example function
targets = np.array(llh_vals)
# ---- Flatten across runs for training ----
# features_flat = features.reshape(-1, num_params)  # (500000, 3)
targets_flat = targets.reshape(-1, 1)             # (500000, 1)

print(targets_flat.shape)
print(features.shape)

# ---- Convert to tensors ----
X_tensor = torch.tensor(features, dtype=torch.float32)
y_tensor = torch.tensor(targets_flat, dtype=torch.float32)

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
    
model = SimpleNet()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# ---- Training Loop ----
epochs = 500
batch_size = 64
for epoch in range(epochs):
    permutation = torch.randperm(X_tensor.size(0))
    epoch_loss = 0.0
    for i in range(0, X_tensor.size(0), batch_size):
        indices = permutation[i:i+batch_size]
        batch_X, batch_y = X_tensor[indices], y_tensor[indices]
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item() * batch_X.size(0)
    if (epoch+1) % 25 == 0 or epoch == 0:
        print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss/X_tensor.size(0):.6f}")

param_ranges = [np.linspace(-3, 3, 100) for _ in range(3)]  
X, Y, Z = np.meshgrid(param_ranges[0], param_ranges[1], param_ranges[2], indexing='ij')

# Step 1: Prepare the grid points as input to the network
features = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1)  # shape (1000000, 3)
features_tensor = torch.tensor(features, dtype=torch.float32)

# Step 2: Run the model on all grid points (in batches if needed)
with torch.no_grad():
    preds = model(features_tensor).cpu().numpy().reshape(100, 100, 100)

print(preds.shape)
with open('predictions-mcmc', 'wb') as f: 
    pickle.dump(preds, f)

# fig, ax = plt.subplots(figsize=(7, 5))
# cax = None

# def update(z_idx):
#     global cax
#     ax.clear()
#     X_slice = X[:, :, z_idx]
#     Y_slice = Y[:, :, z_idx]
#     pred_slice = preds[:, :, z_idx]
#     cax = ax.pcolormesh(X_slice, Y_slice, pred_slice, shading='auto', cmap='viridis', vmin=preds.min(), vmax=preds.max())
#     ax.set_xlabel('Parameter 1 (X)')
#     ax.set_ylabel('Parameter 2 (Y)')
#     ax.set_title(f'NNet Output at Z ≈ {param_ranges[2][z_idx]:.2f}')

# ani = animation.FuncAnimation(
#     fig, update, frames=range(0, preds.shape[2], 2), interval=100, repeat=True
# )

# fig.colorbar(cax, ax=ax, label='NNet Prediction')
# plt.tight_layout()
# plt.show()
# # # Step 3: Pick a Z slice to plot, e.g., Z closest to 0
# # z_value = 1.7
# # z_index = np.abs(param_ranges[2] - z_value).argmin()

# # X_slice = X[:, :, z_index]
# # Y_slice = Y[:, :, z_index]
# # pred_slice = preds[:, :, z_index]

# # plt.figure(figsize=(7, 5))
# # plt.pcolormesh(X_slice, Y_slice, pred_slice, shading='auto', cmap='viridis')
# # plt.xlabel('Parameter 1 (X)')
# # plt.ylabel('Parameter 2 (Y)')
# # plt.title(f'NNet Output at Z ≈ {param_ranges[2][z_index]:.2f}')
# # plt.colorbar(label='NNet Prediction')
# # plt.tight_layout()
# # plt.show()


# # # Predict using trained model
# # with torch.no_grad():
# #     run_pred = model(torch.tensor(run_features, dtype=torch.float32)).numpy().flatten()
# #     run_true = run_targets.flatten()

# # # Example: plot prediction vs true for the first two parameters, fixing the third (if grid-like), or just scatter
# # plt.figure(figsize=(10,5))
# # plt.subplot(1,2,1)
# # plt.title("True Likelihood (first run)")
# # sc = plt.scatter(run_features[:,0], run_features[:,1], c=run_true, cmap='viridis')
# # plt.xlabel('Param 1')
# # plt.ylabel('Param 2')
# # plt.colorbar(sc)

# # plt.subplot(1,2,2)
# # plt.title("Model Prediction (first run)")
# # sc2 = plt.scatter(run_features[:,0], run_features[:,1], c=run_pred, cmap='viridis')
# # plt.xlabel('Param 1')
# # plt.ylabel('Param 2')
# # plt.colorbar(sc2)

# # plt.tight_layout()
# # plt.show()