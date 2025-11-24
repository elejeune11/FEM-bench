def solve_linear_elastic_frame_3D(node_coords: np.ndarray, elements: Sequence[dict], boundary_conditions: dict[int, Sequence[int]], nodal_loads: dict[int, Sequence[float]]):
    n_nodes = node_coords.shape[0]
    K_global = assemble_global_stiffness_matrix_linear_elastic_3D(node_coords, elements)
    P_global = assemble_global_load_vector_linear_elastic_3D(nodal_loads, n_nodes)
    (fixed, free) = partition_degrees_of_freedom(boundary_conditions, n_nodes)
    (u, r) = linear_solve(P_global, K_global, fixed, free)
    return (u, r)