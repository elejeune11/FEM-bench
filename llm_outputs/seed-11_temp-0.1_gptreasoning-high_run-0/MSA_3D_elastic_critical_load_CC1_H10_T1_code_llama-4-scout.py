def MSA_3D_elastic_critical_load_CC1_H10_T1(node_coords: np.ndarray, elements: Sequence[Dict], boundary_conditions: Dict[int, Sequence[int]], nodal_loads: Dict[int, Sequence[float]]):
    K_e_global = assemble_global_stiffness_matrix_linear_elastic_3D(node_coords, elements)
    P_global = assemble_global_load_vector_linear_elastic_3D(nodal_loads, node_coords.shape[0])
    (fixed, free) = partition_degrees_of_freedom(boundary_conditions, node_coords.shape[0])
    (u_global, _) = linear_solve(P_global, K_e_global, fixed, free)
    K_g_global = assemble_global_geometric_stiffness_3D_beam(node_coords, elements, u_global)
    (elastic_critical_load_factor, deformed_shape_vector) = eigenvalue_analysis(K_e_global, K_g_global, boundary_conditions, node_coords.shape[0])
    return (elastic_critical_load_factor, deformed_shape_vector)