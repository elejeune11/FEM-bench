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
    num_nodes = int(num_elements) + 1
    node_coords = np.linspace(x_min, x_max, num_nodes)
    left_nodes = np.arange(num_elements, dtype=int)
    right_nodes = np.arange(1, num_elements + 1, dtype=int)
    element_connectivity = np.vstack((left_nodes, right_nodes)).T
    return (node_coords, element_connectivity)