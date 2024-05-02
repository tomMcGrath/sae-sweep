import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Callable


class SparseAutoEncoder(torch.nn.Module):
    """A standard Sparse Autoencoder as defined in 'Towards Monosemanticity'"""

    def __init__(
            self,
            model_dim,
            feature_dim: int,
            has_encoder_bias: bool = True,
            has_decoder_bias: bool = True,
            input_noise: bool = False,
            input_noise_scale: float = 0.,
            ):
        super().__init__()
        self.model_dim = model_dim
        self.feature_dim = feature_dim
        self.input_noise = input_noise
        if self.input_noise:
            self.input_noise_scale = input_noise_scale

        # Create & initialise layers
        # TODO(tomMcGrath): add optional initializers
        self.encoder = torch.nn.Linear(model_dim, feature_dim, bias=has_encoder_bias)
        self.decoder = torch.nn.Linear(feature_dim, model_dim, bias=has_decoder_bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.input_noise:
            input_noise = torch.normal(0, self.input_noise_scale * torch.ones_like(x))
            x += input_noise

        x = self.encoder(x)
        f = F.relu(x)
        return {'x_reconstruct': self.decoder(f), 'features': f}
    
    def get_decoder_norms(self) -> torch.Tensor:
        """Return a vector where entry i is ||W_{d,i}||_2."""
        return torch.linalg.vector_norm(self.decoder.weight, dim=0)
