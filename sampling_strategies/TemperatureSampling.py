from .SamplingStrategy import SamplingStrategy
from torch import Tensor
import torch
from random import randint


class TemperatureSampling(SamplingStrategy):

    def __init__(self, temperature: float):
        super(TemperatureSampling, self).__init__()
        self.temp = temperature
    

    def get_next_token_id(self,probabilities:Tensor)->Tensor:
        """
        Perform temperature sampling on the given probability distribution.
        
        Parameters:
        - probabilities (torch.Tensor): The probability distribution over the vocabulary.
        - temperature (float): The temperature to apply. Lower values (< 1) make the distribution peakier,
        higher values (> 1) make it more flat.
        
        Returns:
        - int: The index of the sampled token.
        """
        if self.temp <= 0:
            raise ValueError("Temperature must be greater than zero")
                
        # Apply temperature to the probabilities
        adjusted_probs = torch.pow(probabilities, 1.0 / self.temp)
        
        # Normalize the probabilities
        adjusted_probs = adjusted_probs / torch.sum(adjusted_probs)
        
        # Sample from the adjusted probability distribution
        token_index= torch.multinomial(adjusted_probs, 1)

        token_index=torch.unsqueeze(token_index,0)

        return token_index
    
    def __repr__(self):
        return(f'Temeprature sampling with temperature= {self.temp}')
