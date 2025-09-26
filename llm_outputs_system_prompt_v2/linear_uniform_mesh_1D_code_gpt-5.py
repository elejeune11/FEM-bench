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
    if not isinstance(num_elements, int):
        raise TypeError('num_elements must be an integer.')
    if num_elements < 1:
        raise ValueError('num_elements must be >= 1.')
    try:
        x_min_val = float(x_min)
        x_max_val = float(x_max)
    except Exception as e:
        raise TypeError('x_min and x_max must be numeric.') from e
    if not (np.isfinite(x_min_val) and np.isfinite(x_max_val)):
        raise ValueError('x_min and x_max must be finite numbers.')
    if x_max_val <= x_min_val:
        raise ValueError('x_max must be greater than x_min.')
    num_nodes = num_elements + 1
    node_coords = np.linspace(x_min_val, x_max_val, num_nodes, dtype=float)
    element_connectivity = np.column_stack((np.arange(num_elements, dtype=int), np.arange(1, num_elements + 1, dtype=int)))
    return (node_coords, element_connectivity)