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
    if not np.isscalar(num_elements):
        raise TypeError('num_elements must be a scalar.')
    n = int(num_elements)
    if n != num_elements or n < 1:
        raise ValueError('num_elements must be a positive integer.')
    if not (np.isscalar(x_min) and np.isscalar(x_max)):
        raise TypeError('x_min and x_max must be scalars.')
    if not (np.isfinite(x_min) and np.isfinite(x_max)):
        raise ValueError('x_min and x_max must be finite.')
    if x_max < x_min:
        raise ValueError('x_max must be greater than or equal to x_min.')
    num_nodes = n + 1
    node_coords = np.linspace(float(x_min), float(x_max), num_nodes)
    start_nodes = np.arange(n, dtype=int)
    end_nodes = start_nodes + 1
    element_connectivity = np.stack((start_nodes, end_nodes), axis=1)
    return (node_coords, element_connectivity)