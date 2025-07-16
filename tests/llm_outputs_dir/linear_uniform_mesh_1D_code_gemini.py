import numpy as np

def linear_uniform_mesh_1D(x_min: float, x_max: float, num_elements: int) -> np.ndarray:
    """
    Generate a 1D linear mesh.

    Parameters:
        x_min (float): Start coordinate of the domain.
        x_max (float): End coordinate of the domain.
        num_elements (int): Number of linear elements.

    Returns:
        np.ndarray[np.ndarray, np.ndarray]:
            - node_coords: 1D array of node coordinates (shape: [num_nodes])
            - element_connectivity: 2D array of element connectivity 
              (shape: [num_elements, 2]) with node indices per element
    """
    num_nodes = num_elements + 1
    node_coords = np.linspace(x_min, x_max, num_nodes)
    element_connectivity = np.zeros((num_elements, 2), dtype=int)
    for i in range(num_elements):
        element_connectivity[i, 0] = i
        element_connectivity[i, 1] = i + 1
    return node_coords, element_connectivity