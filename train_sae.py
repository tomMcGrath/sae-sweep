import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, RandomSampler
from tqdm import tqdm

import models
import activations_dataset


data_paths = [
    'activations/layers-16-0.pt',
    'activations/layers-16-1.pt',
    'activations/layers-16-2.pt',
    'activations/layers-16-3.pt',
    'activations/layers-16-4.pt',
    'activations/layers-16-5.pt',
    'activations/layers-16-6.pt',
    'activations/layers-16-7.pt',
]
print('Loading activations data')
dataset = activations_dataset.ActivationsDataset(data_paths)
print(f'Loaded {len(dataset):.2e} activations.')

bsz = 2048
loader = DataLoader(
    dataset,
    batch_size=bsz,
    sampler=RandomSampler(dataset),
)

print('Building model')
d_model = 4096
expansion_factor = 32
n_features = d_model * expansion_factor
device = 'cuda:1'
sae = models.SparseAutoEncoder(d_model, n_features)
sae.to(device)
lr = 1e-5
beta1 = 0.9
beta2 = 0.99
optimizer = optim.Adam(sae.parameters(), lr=lr, betas=[beta1, beta2])
mse_loss = nn.MSELoss()
l1_multiplier = 5.

print('Beginning SAE training')
for x in tqdm(loader):
    # Predictions
    acts = x['activations'].float().to(device)
    sae_outputs = sae(acts)

    # Reconstruction loss
    x_reconstruct = sae_outputs['x_reconstruct']
    recon_loss = mse_loss(acts, x_reconstruct)

    # L_1 loss
    feats = sae_outputs['features']
    l1_loss = torch.linalg.vector_norm(feats, ord=1, dim=1)
    l1_loss = torch.mean(l1_loss)

    # Overall loss
    loss = recon_loss + l1_multiplier * l1_loss

    # Additional metrics
    l0 = torch.linalg.vector_norm(feats, ord=0, dim=1)
    l0 = torch.mean(l0)
    # TODO(tomMcGrath): add dead neuron count

    # Gradient step
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    print(f'Total loss: {loss:.3f}, Reconstruction loss {recon_loss:.3f}, L1 loss {l1_loss:.3f}, L0 loss: {l0:.3f}')
