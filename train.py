import numpy as np
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt 

with open('loglik_grid_parallel-120.pkl', 'rb') as f:
    loglik_grid = pickle.load(f)

loglik_grid = np.array(loglik_grid)
likelihood_grid = np.exp(loglik_grid)
print(loglik_grid.shape)
param_ranges = [np.linspace(-3, 3, 100) for _ in range(3)]  
X, Y, Z = np.meshgrid(param_ranges[0], param_ranges[1], param_ranges[2], indexing='ij')

features = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1)  
targets = likelihood_grid.ravel()[:, None]                       

print("features shape:", features.shape)
print("targets shape:", targets.shape)

X_tensor = torch.tensor(features, dtype=torch.float32)
y_tensor = torch.tensor(targets, dtype=torch.float32)

class SingleLayerNet(nn.Module):
    def __init__(self, input_size, hidden, output_size):
        super(SingleLayerNet, self).__init__()
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

model = SingleLayerNet(input_size=3, hidden=64, output_size=1)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

epochs = 100
batch_size = 10000  

for epoch in range(epochs):
    permutation = torch.randperm(X_tensor.size()[0])
    epoch_loss = 0.0
    for i in range(0, X_tensor.size()[0], batch_size):
        indices = permutation[i:i+batch_size]
        batch_X, batch_y = X_tensor[indices], y_tensor[indices]
        
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item() * batch_X.size(0)
    if (epoch + 1) % 50 == 0 or epoch == 0:
        print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss/X_tensor.size()[0]:.6f}")

torch.save(model.state_dict(), "single_layer_net_3dgrid.pt")

z_index = np.abs(param_ranges[2][10])

X_slice = X[:, :, z_index]
Y_slice = Y[:, :, z_index]
inputs_slice = np.stack([X_slice.ravel(), Y_slice.ravel(), np.full(X_slice.size, param_ranges[2][z_index])], axis=1)
inputs_slice_tensor = torch.tensor(inputs_slice, dtype=torch.float32)

with torch.no_grad():
    pred_slice = model(inputs_slice_tensor).numpy().reshape(X_slice.shape)
    true_slice = likelihood_grid[:, :, z_index]

# Plot
plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
plt.title("model pred")
plt.pcolormesh(X_slice, Y_slice, pred_slice, shading='auto')
plt.xlabel('X')
plt.ylabel('Y')
plt.colorbar(label='pred')

plt.subplot(1,2,2)
plt.title("true llh")
plt.pcolormesh(X_slice, Y_slice, true_slice, shading='auto')
plt.xlabel('X')
plt.ylabel('Y')
plt.colorbar(label='true')

plt.tight_layout()
plt.show()