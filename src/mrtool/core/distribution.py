"""
    distribution
    ~~~~~~~~~~~~
"""
from typing import ClassVar, Iterable, Union
from dataclasses import dataclass
import numpy as np


@dataclass
class Distribution:
    default_params: ClassVar[np.ndarray] = np.array([])
    name: ClassVar[str] = 'generic'
    params: Union[np.ndarray, Iterable] = None

    def __post_init__(self):
        self.num_params = len(self.default_params)

        if self.params is None:
            self.params = np.array([self.default_params])

        if not isinstance(self.params, np.ndarray):
            self.params = np.asarray(self.params)

        if self.params.ndim == 1:
            self.params = self.params[None, :]

        if self.params.shape[1] != self.num_params:
            raise ValueError(f"Parameter size ({self.params.shape[1]}) for each dimension"
                             f"must be consistent with default_params ({self.num_params})")

        self.dim = len(self.params)

    def is_informative(self) -> bool:
        raise NotImplementedError()

    def is_identical(self) -> bool:
        if not self.is_informative():
            return True
        else:
            unique_params = np.unique(self.params, axis=0)
            return len(unique_params) == 1

    def change_dim(self, dim: int):
        """Change the dimension of the distribution.
        """
        if dim < 1:
            raise ValueError(f"dim = {dim} has to be at least 1.")

        if dim != self.dim:
            if not self.is_identical():
                raise ValueError("In order to change dimension, distribution has to be identical"
                                 "(include uninformative) across dimension.")

            if dim < self.dim:
                self.params = self.params[:dim]
            else:
                self.params = np.repeat(self.params,
                                        [1]*(self.dim - 1) + [dim - self.dim + 1], axis=0)

            self.__post_init__()


@dataclass
class Gaussian(Distribution):
    default_params: ClassVar[np.ndarray] = np.array([0.0, np.inf])
    name: ClassVar[str] = 'Gaussian'

    def __post_init__(self):
        super().__post_init__()
        self.mean = self.params[:, 0]
        self.std = self.params[:, 1]
        if not all(self.std > 0.0):
            raise ValueError("Gaussian distribution standard errors have to be positive.")

    def is_informative(self) -> bool:
        return not all(np.isinf(self.std))


@dataclass
class Laplace(Distribution):
    default_params: ClassVar[np.ndarray] = np.array([0.0, np.inf])
    name: ClassVar[str] = 'Laplace'

    def __post_init__(self):
        super().__post_init__()
        self.mean = self.params[:, 0]
        self.std = self.params[:, 1]
        if not np.all(self.std > 0.0):
            raise ValueError("Laplace distribution standard errors have to be positive.")

    def is_informative(self) -> bool:
        return not all(np.isinf(self.std))


@dataclass
class Uniform(Distribution):
    default_params: ClassVar[np.ndarray] = np.array([-np.inf, np.inf])
    name: ClassVar[str] = 'Uniform'

    def __post_init__(self):
        super().__post_init__()
        self.lb = self.params[:, 0]
        self.ub = self.params[:, 1]
        if not np.all(self.lb <= self.ub):
            raise ValueError("Uniform distribution lower bounds have to be less or equal than"
                             "upper bounds.")

        if np.any(np.isneginf(self.lb) & np.isneginf(self.ub)) or np.any(np.isposinf(self.lb) & np.isposinf(self.ub)):
            raise ValueError("Uniform distribution lower bounds and upper bounds cannot be both"
                             "positive infinity or negative infinity at the same time.")

    def is_informative(self) -> bool:
        return not np.all(np.isneginf(self.lb) & np.isposinf(self.ub))
