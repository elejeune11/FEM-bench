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
    if not isinstance(num_elements, (int, np.integer)):
        raise TypeError('num_elements must be an integer')
    if num_elements < 1:
        raise ValueError('num_elements must be a positive integer')
    if not np.isfinite(x_min) or not np.isfinite(x_max):
        raise ValueError('x_min and x_max must be finite numbers')
    if x_max <= x_min:
        raise ValueError('x_max must be greater than x_min')
    node_coords = np.linspace(x_min, x_max, num_elements + 1, dtype=float)
    left_nodes = np.arange(0, num_elements, dtype=int)
    right_nodes = left_nodes + 1
    element_connectivity = np.column_stack((left_nodes, right_nodes))
    return (node_coords, element_connectivity)