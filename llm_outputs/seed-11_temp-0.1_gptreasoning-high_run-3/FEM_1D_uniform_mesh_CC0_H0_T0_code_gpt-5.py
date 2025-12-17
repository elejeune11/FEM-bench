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
    if not isinstance(num_elements, (int, np.integer)) or isinstance(num_elements, bool):
        raise TypeError('num_elements must be an integer.')
    if num_elements < 1:
        raise ValueError('num_elements must be >= 1.')
    x_min_val = float(x_min)
    x_max_val = float(x_max)
    if x_max_val < x_min_val:
        raise ValueError('x_max must be greater than or equal to x_min.')
    node_coords = np.linspace(x_min_val, x_max_val, num_elements + 1, dtype=float)
    left_nodes = np.arange(0, num_elements, dtype=int)
    right_nodes = left_nodes + 1
    element_connectivity = np.stack((left_nodes, right_nodes), axis=1)
    return (node_coords, element_connectivity)