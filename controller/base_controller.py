from abc import ABC, abstractmethod
from torch import Tensor

class BaseController(ABC):

    @abstractmethod
    def __call__(self, raw_policy_output: Tensor) -> Tensor:
        """Map raw policy output -> action tensor expected by dynamics."""
        ...