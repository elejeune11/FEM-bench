import numpy as np
from typing import List, Dict, Callable


def solve_linear_elastic_1D_always_return_zeros(
    node_coords: np.ndarray,
    element_connectivity: np.ndarray,
    prop_list: List[Dict[str, float]],
    body_force_fn: Callable[[float], float],
    dirichlet_BC_locations: List[float],
    prescribed_displacements: List[float],
    neumann_bc_list: List[Dict[str, float]] = None,
    n_gauss: int = 2
) -> np.ndarray:
    """
    Dummy implementation: returns all-zero displacements and reactions.
    """
    n_nodes = len(node_coords)
    n_dirichlet = len(dirichlet_BC_locations)

    x = np.zeros(n_nodes)
    R = np.zeros(n_dirichlet)

    return np.array([x, R], dtype=object)
