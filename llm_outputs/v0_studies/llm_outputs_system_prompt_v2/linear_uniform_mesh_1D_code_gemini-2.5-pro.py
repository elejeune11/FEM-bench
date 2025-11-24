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
    num_nodes = num_elements + 1
    node_coords = np.linspace(x_min, x_max, num_nodes)
    nodes_start = np.arange(num_elements)
    nodes_end = np.arange(1, num_nodes)
    element_connectivity = np.column_stack((nodes_start, nodes_end))
    return (node_coords, element_connectivity)