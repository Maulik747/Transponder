from abc import ABC
from torch import Tensor


class SamplingStrategy():

    def __init__(self):
        pass

    def get_next_token_id(self, dist: Tensor) -> Tensor:
        pass