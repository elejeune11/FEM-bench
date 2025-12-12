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
    if not isinstance(num_elements, (int, np.integer)):
        raise TypeError('num_elements must be an integer')
    if num_elements < 1:
        raise ValueError('num_elements must be at least 1')
    x_min = float(x_min)
    x_max = float(x_max)
    if not np.isfinite(x_min) or not np.isfinite(x_max):
        raise ValueError('x_min and x_max must be finite numbers')
    if x_max <= x_min:
        raise ValueError('x_max must be greater than x_min')
    num_nodes = num_elements + 1
    node_coords = np.linspace(x_min, x_max, num_nodes)
    left = np.arange(num_elements, dtype=int)
    right = left + 1
    element_connectivity = np.stack((left, right), axis=1)
    return (node_coords, element_connectivity)