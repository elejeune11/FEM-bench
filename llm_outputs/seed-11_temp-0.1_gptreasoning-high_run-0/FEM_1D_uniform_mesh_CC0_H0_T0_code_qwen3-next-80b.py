def FEM_1D_uniform_mesh_CC0_H0_T0(x_min: float, x_max: float, num_elements: int) -> np.ndarray:
    num_nodes = num_elements + 1
    node_coords = np.linspace(x_min, x_max, num_nodes)
    element_connectivity = np.zeros((num_elements, 2), dtype=int)
    for i in range(num_elements):
        element_connectivity[i, 0] = i
        element_connectivity[i, 1] = i + 1
    return (node_coords, element_connectivity)