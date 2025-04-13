from .SamplingStrategy import SamplingStrategy
from torch import Tensor
from random import randint
import torch

class TopKSamplingStrategy(SamplingStrategy):

    def __init__(self, k: int):
        super(TopKSamplingStrategy, self).__init__()
        self.k = k

    def get_next_token_id(self, dist: Tensor) -> Tensor:
        super(TopKSamplingStrategy, self).get_next_token_id(dist)
        indices = dist.sort(dim=0).indices
        top_k_indices = indices[-1 * self.k :]
        random_k = randint(0, self.k - 1)
        next_id=torch.unsqueeze(top_k_indices[random_k],0)
        next_id=torch.unsqueeze(next_id,0)
        return next_id
