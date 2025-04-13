from abc import ABC, abstractmethod
from torch import Tensor


class PonderingCriteria(ABC):

    def __init__(self):
        pass

    @abstractmethod
    def pondering_needed(self, dist: Tensor) -> bool:
        pass
