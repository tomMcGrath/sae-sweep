import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, RandomSampler
from tqdm import tqdm
import wandb

import models
import activations_dataset


def main():
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

    # torch.set_float32_matmul_precision('high')

    device = 'cuda:1'
    bsz = wandb.config.batch_size
    loader = DataLoader(
        dataset,
        batch_size=bsz,
        sampler=RandomSampler(dataset),
        pin_memory=True,
        pin_memory_device=device,
    )

    print('Building model')
    d_model = 4096
    expansion_factor = wandb.config.expansion_factor
    n_features = d_model * expansion_factor
    
    sae = models.SparseAutoEncoder(
        d_model,
        n_features,
        input_noise=wandb.config.input_noise,
        input_noise_scale=wandb.config.input_noise_scale,
        )
    sae.to(device)
    sae = sae.bfloat16()
    sae = torch.compile(sae)
    lr = wandb.config.learning_rate
    beta1 = wandb.config.adam_beta1
    beta2 = wandb.config.adam_beta2
    l1_multiplier = wandb.config.l1_multiplier

    optimizer = optim.Adam(sae.parameters(), lr=lr, betas=[beta1, beta2])
    mse_loss = nn.MSELoss()

    print('Beginning SAE training')
    tensorboard_trace_handler = torch.profiler.tensorboard_trace_handler('traces/')
    with torch.profiler.profile(
        schedule=torch.profiler.schedule(
            wait=4,
            warmup=2,
            active=6,
            repeat=1
        ),
        on_trace_ready=tensorboard_trace_handler,
        with_stack=True,
    ) as profiler:
        for x in tqdm(loader):
            # Predictions
            # acts = x['activations'].float().to(device)
            acts = x['activations'].to(device)
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

            # Logging
            wandb.log({
                'total_loss': loss,
                'recon_loss': recon_loss,
                'l1_loss': l1_loss,
                'weighted_l1_loss': l1_loss * l1_multiplier,
                'l0_loss': l0,
            })

            # Gradient step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            profiler.step()


if __name__ == '__main__':
    config = {
        'expansion_factor': 16,
        'batch_size': 8192,
        'learning_rate': 1e-5,
        'adam_beta1': 0.9,
        'adam_beta2': 0.99,
        'l1_multiplier': 1.,
        'input_noise': False,
        'input_noise_scale': 0.,
    }
    run = wandb.init(project='sae-sweep-prototyping', config=config)
    main()