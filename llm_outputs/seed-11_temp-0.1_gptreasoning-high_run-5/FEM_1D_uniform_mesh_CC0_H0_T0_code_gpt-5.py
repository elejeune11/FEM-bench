def FEM_1D_uniform_mesh_CC0_H0_T0(x_min: float, x_max: float, num_elements: int) -> np.ndarray:
    """
    Generate a 1D linear uniform mesh with evenly spaced nodes.
    Parameters:
        x_min (float): Start coordinate of the domain.
        x_max (float): End coordinate of the domain.
        num_elements (int): Number of linear elements.
    Returns:
        tuple[np.ndarray, np.ndarray]:
            node_coords: 1D array of node coordinates (shape: [num_nodes]).
            element_connectivity: 2D array of element connectivity 
                (shape: [num_elements, 2]), where each row lists the two node indices of an element.
    Notes:
    """
    import numpy as np
    try:
        x0 = float(x_min)
        x1 = float(x_max)
    except (TypeError, ValueError):
        raise TypeError('x_min and x_max must be real numbers')
    if not np.isfinite(x0) or not np.isfinite(x1):
        raise ValueError('x_min and x_max must be finite numbers')
    if x1 <= x0:
        raise ValueError('x_max must be greater than x_min')
    if isinstance(num_elements, bool):
        raise TypeError('num_elements must be a positive integer')
    if isinstance(num_elements, (int, np.integer)):
        ne = int(num_elements)
    elif isinstance(num_elements, float) and num_elements.is_integer():
        ne = int(num_elements)
    else:
        raise TypeError('num_elements must be a positive integer')
    if ne <= 0:
        raise ValueError('num_elements must be a positive integer')
    node_coords = np.linspace(x0, x1, ne + 1, dtype=float)
    element_connectivity = np.column_stack((np.arange(ne, dtype=int), np.arange(1, ne + 1, dtype=int)))
    return (node_coords, element_connectivity)