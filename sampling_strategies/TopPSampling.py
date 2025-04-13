
from .SamplingStrategy import SamplingStrategy
from torch import Tensor
import torch
from random import randint
class TopPSamplingStrategy(SamplingStrategy):
    def __init__(self, p: float):
        super(TopPSamplingStrategy, self).__init__()
        self.p = p
    def get_next_token_id(self,probabilities:Tensor)->Tensor:
        """
        Apply top-p (nucleus) sampling to a probability distribution.
        Args:
            probabilities (torch.Tensor): The probability distribution tensor.
            p (float): The cumulative probability threshold.
        Returns:
            int: The index of the selected item.
        """
        # Ensure probabilities is a torch tensor
        probabilities=probabilities[0]##turning it into a single dimension
        probabilities = torch.tensor(probabilities) if not isinstance(probabilities, torch.Tensor) else probabilities
        # Sort the probabilities and get their cumulative sum
        sorted_probabilities, sorted_indices = torch.sort(probabilities, descending=True)
        cumulative_probabilities = torch.cumsum(sorted_probabilities, dim=0)
        # Find the cutoff index where the cumulative sum exceeds p

        cutoff_index = torch.searchsorted(cumulative_probabilities, self.p)
        # Create a mask to filter out the top-p probabilities
        mask = cumulative_probabilities <= cumulative_probabilities[cutoff_index - 1]
        # Normalize the filtered probabilities
        top_p_probabilities = sorted_probabilities[mask]
        top_p_probabilities /= top_p_probabilities.sum()
        # Sample from the filtered and normalized distribution
        sampled_index = torch.multinomial(top_p_probabilities, 1).item()
        # Map the sampled index back to the original index
        next_id=torch.unsqueeze(sorted_indices[mask][sampled_index],0)
        return torch.unsqueeze(next_id,0)
