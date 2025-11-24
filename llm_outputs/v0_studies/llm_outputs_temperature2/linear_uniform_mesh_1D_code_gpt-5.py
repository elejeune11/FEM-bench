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
    if not (np.isfinite(x_min_f) and np.isfinite(x_max_f)):
        raise ValueError('x_min and x_max must be finite numbers')
    num_nodes = num_elements + 1
    node_coords = np.linspace(x_min_f, x_max_f, num_nodes)
    starts = np.arange(num_elements, dtype=int)
    ends = starts + 1
    element_connectivity = np.column_stack((starts, ends))
    return (node_coords, element_connectivity)