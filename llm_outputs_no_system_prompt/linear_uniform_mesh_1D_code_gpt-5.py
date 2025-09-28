def linear_uniform_mesh_1D(x_min: float, x_max: float, num_elements: int) -> np.ndarray:
    """
    Generate a 1D linear mesh.
    Parameters:
        x_min (float): Start coordinate of the domain.
        x_max (float): End coordinate of the domain.
        num_elements (int): Number of linear elements.
    Returns:
        (node_coords, element_connectivity):
              (shape: [num_elements, 2]) with node indices per element
    """
    import numpy as np
    import pytest
    if not isinstance(num_elements, (int, np.integer)):
        raise TypeError('num_elements must be an integer')
    if num_elements < 1:
        raise ValueError('num_elements must be >= 1')
    try:
        x_min_f = float(x_min)
        x_max_f = float(x_max)
    except (TypeError, ValueError):
        raise TypeError('x_min and x_max must be real numbers')
    num_nodes = num_elements + 1
    node_coords = np.linspace(x_min_f, x_max_f, num_nodes, dtype=float)
    left_nodes = np.arange(num_elements, dtype=int)
    right_nodes = left_nodes + 1
    element_connectivity = np.stack((left_nodes, right_nodes), axis=1)
    return (node_coords, element_connectivity)