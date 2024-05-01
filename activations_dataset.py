import torch
from tqdm import tqdm


class ActivationsDataset(torch.utils.data.Dataset):

    def __init__(self, tensor_paths):
        self.activations = []
        self.tokens = []

        # Load data
        # TODO(tomMcGrath): drop <PAD> and <BOS> tokens on loading
        for path in tqdm(tensor_paths):
            data = torch.load(path)
            self.activations.append(data['activations'].flatten(0, 1))
            self.tokens.append(data['tokens'].flatten(0))

        # Consolidate data
        self.activations = torch.cat(self.activations)
        self.tokens = torch.cat(self.tokens)

    def __getitem__(self, idx):
        return {
            'activations': self.activations[idx],
            'tokens': self.tokens[idx],
            }
    
    def __len__(self):
        return self.activations.shape[0]
