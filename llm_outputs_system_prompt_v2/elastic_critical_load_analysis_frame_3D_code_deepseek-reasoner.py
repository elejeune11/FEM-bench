def elastic_critical_load_analysis_frame_3D(node_coords: np.ndarray, elements: Sequence[dict], boundary_conditions: dict[int, Sequence[int]], nodal_loads: dict[int, Sequence[float]]):
    n_nodes = node_coords.shape[0]
    bc_converted = {}
    for (node, spec) in boundary_conditions.items():
        if len(spec) == 6 and all((isinstance(x, bool) for x in spec)):
            bc_converted[node] = spec
        else:
            bool_list = [False] * 6
            for dof_index in spec:
                bool_list[dof_index] = True
            bc_converted[node] = bool_list
    K_global = assemble_global_stiffness_matrix_linear_elastic_3D(node_coords, elements)
    P_global = assemble_global_load_vector_linear_elastic_3D(nodal_loads, n_nodes)
    (fixed, free) = partition_degrees_of_freedom(bc_converted, n_nodes)
    (u_global, _) = linear_solve(P_global, K_global, fixed, free)
    K_g_global = assemble_global_geometric_stiffness_3D_beam(node_coords, elements, u_global)
    (elastic_critical_load_factor, deformed_shape_vector) = eigenvalue_analysis(K_global, K_g_global, bc_converted, n_nodes)
    return (elastic_critical_load_factor, deformed_shape_vector)