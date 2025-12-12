def MSA_3D_elastic_critical_load_CC1_H10_T1(node_coords: np.ndarray, elements: Sequence[dict], boundary_conditions: dict[int, Sequence[int]], nodal_loads: dict[int, Sequence[float]]):
    n_nodes = node_coords.shape[0]
    K_e_global = assemble_global_stiffness_matrix_linear_elastic_3D(node_coords, elements)
    P_global = assemble_global_load_vector_linear_elastic_3D(nodal_loads, n_nodes)
    (fixed, free) = partition_degrees_of_freedom(boundary_conditions, n_nodes)
    (u_global, _) = linear_solve(P_global, K_e_global, fixed, free)
    for ele in elements:
        ele['I_rho'] = ele['I_y'] + ele['I_z']
    K_g_global = assemble_global_geometric_stiffness_3D_beam(node_coords, elements, u_global)
    (elastic_critical_load_factor, deformed_shape_vector) = eigenvalue_analysis(K_e_global, K_g_global, boundary_conditions, n_nodes)
    return (elastic_critical_load_factor, deformed_shape_vector)