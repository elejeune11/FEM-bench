def elastic_critical_load_analysis_frame_3D(node_coords: np.ndarray, elements: Sequence[dict], boundary_conditions: dict[int, Sequence[int]], nodal_loads: dict[int, Sequence[float]]):
    K_e_global = assemble_global_stiffness_matrix_linear_elastic_3D(node_coords, elements)
    n_nodes = node_coords.shape[0]
    P_global = assemble_global_load_vector_linear_elastic_3D(nodal_loads, n_nodes)
    (fixed, free) = partition_degrees_of_freedom(boundary_conditions, n_nodes)
    (u_global, _) = linear_solve(P_global, K_e_global, fixed, free)
    K_g_global = assemble_global_geometric_stiffness_3D_beam(node_coords, elements, u_global)
    (elastic_critical_load_factor, deformed_shape_vector) = eigenvalue_analysis(K_e_global, K_g_global, boundary_conditions, n_nodes)
    return (elastic_critical_load_factor, deformed_shape_vector)