import torch
import torch.nn as nn

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

class EmbeddingLayer(nn.Module):
    def __init__(self, num_nodes, embedding_dim, seed=None):
        super(EmbeddingLayer, self).__init__()
        if seed is not None:
            set_seed(seed)
        self.embedding = nn.Embedding(num_nodes, embedding_dim)
        for param in self.embedding.parameters():
            param.requires_grad = False

    def forward(self, x):
        return self.embedding(x)