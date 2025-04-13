from .PonderingCriteria import PonderingCriteria
import torch


class EntropyPonderingCriteria(PonderingCriteria):

    def __init__(self, threshold: float):
        self.threshold = threshold
        super(EntropyPonderingCriteria, self).__init__()

    def pondering_needed(self, dist: torch.Tensor):
        super(EntropyPonderingCriteria, self).pondering_needed(dist)
        entropy = (-1 * dist.log2() * dist).sum().item()
        return entropy > self.threshold
    

    def __repr__(self):  
        return f"Entropy Pondering Criteria with threshold {self.threshold}" 
    
