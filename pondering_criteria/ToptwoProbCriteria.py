from .PonderingCriteria import PonderingCriteria
import torch
import numpy as np


class ToptwoProbCriteria(PonderingCriteria):

    def __init__(self, threshold: float):
        self.threshold = threshold
        super(ToptwoProbCriteria, self).__init__()

    def pondering_needed(self, dist: torch.Tensor):
        super(ToptwoProbCriteria, self).pondering_needed(dist)
        distribution = dist.numpy()

        # Sort the distribution in descending order

        sorted_distribution = np.sort(distribution)
        
        # Get the top two values
        top_two_values = sorted_distribution[0][-2:]
        
        # Calculate the difference
        difference = top_two_values[1] - top_two_values[0]

        top_less=top_two_values[1]<0.5

        return difference<self.threshold and top_less
    
    def __repr__(self):  
        return f"Top two probability Criteria with threshold {self.threshold} " 
