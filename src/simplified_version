# This shows the simplified process of the ML-EMOS at one location and doesn't include weather regimes

import numpy as np
import pandas as pd
import torch
import torchist
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# Establish ML EMOS model
class ML_EMOS(nn.Module):
    def __init__(self):
        super(ML_EMOS, self).__init__()
        self.fc1 = nn.Linear(4, 4)
        self.acti = nn.Sigmoid()

    def forward(self, x):
        x = self.fc1(x)
        x = self.acti(x)
        return x

# Initialize neural network model

model = ML_EMOS()
# Adam optimizer used to update parameters
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Loss function: CRPS
def crps_single(fore, obs):
    fore = torch.sort(fore)[0]  # Sort the forecast values member by member 
    bin_edges = torch.cat((fore, (fore[-1] + 1).reshape(1)))  # Define histogram bins
    pdf = torchist.histogramdd(fore.unsqueeze(-1), edges=bin_edges)
    pdf = torchist.normalize(pdf)[0]  # Normalize and form a probability distribution

    cdf = torch.cumsum(pdf, dim=0)  # Cumulated distribution function
    cdf = torch.sort(cdf)[0]  # sort the cdf

    bin_widths = bin_edges[1:] - bin_edges[:-1]  # calculate the length of every bin 
    obs_h = torch.ge(bin_edges[:-1], obs).int()  # Record the step function
    crps = torch.sum(torch.abs(cdf - obs_h.float()) ** 2 * bin_widths, dim=0)
    return crps
def crps_bunch(fore, obs):
    crps = crps_single(fore[0], obs[0].reshape(1)).reshape(1)
    for i in range(1, len(obs)):
        crps = torch.cat((crps, crps_single(fore[i], obs[i].reshape(1)).reshape(1)))
    return torch.mean(crps)

# Data loads (4-members ensemble forecast over 5 days)
fore = [torch.tensor([2.0, 5.0, 3.0, 4.0]),
        torch.tensor([2.0, 6.0, 4.0, 6.0]),
        torch.tensor([3.0, 4.0, 5.0, 5.0]),
        torch.tensor([1.0, 1.0, 1.5, 1.4]),
        torch.tensor([3.0, 4.0, 3.5, 4.5])]
obs = torch.tensor([3.5, 5.0, 4.5, 1.3, 3.9])

# Split data
train_fore = torch.stack(fore[:3])
val_fore = torch.stack(fore[3:4])
test_fore = torch.stack(fore[4:5])

train_obs = obs[:3].view(-1, 1)
val_obs = obs[3:4].view(-1, 1)
test_obs = obs[4:5].view(-1, 1)

# Lists to store loss values
train_losses = []
val_losses = []

# Train
for epoch in range(20):
    # Start training mode
    model.train()
    output_train = model(train_fore)
    # Initialize loss and grads
    train_loss = crps_bunch(output_train, train_obs)
    optimizer.zero_grad()
    # Keep Calculating loss via backward every time
    # And update grads and parameters
    train_loss.backward()
    optimizer.step()
    # Start validating
    model.eval()
    with torch.no_grad():
        output_val = model(val_fore)
        val_loss = crps_bunch(output_val, val_obs)
    # Store train and val loss values
    train_losses.append(train_loss.detach().numpy())
    val_losses.append(val_loss.detach().numpy())
    # Print process of training and validating
    if (epoch + 1) % 1 == 0:
        print(f'Epoch {epoch + 1}, train CRPS: {train_loss.item()}, val CRPS: {val_loss.item()}')

model.eval()
with torch.no_grad():
    output_test = model(test_fore)
    test_loss = crps_bunch(output_test, test_obs)
    print(f'transformed forecast: {output_test}')
    print(f'CRPS for transformed forecast: {test_loss.item()}')

# Plotting the training and validation CRPS
plt.figure(figsize=(10, 5))
plt.plot(train_losses, label='Train CRPS')
plt.plot(val_losses, label='Val CRPS')
plt.xlabel('Epoch')
plt.ylabel('CRPS')
plt.title('Train and Validation CRPS over Epochs')
plt.legend()
plt.show()

plt.figure(figsize=(10, 5))
width = 0.2
torch.save(model, 'model.pt')
