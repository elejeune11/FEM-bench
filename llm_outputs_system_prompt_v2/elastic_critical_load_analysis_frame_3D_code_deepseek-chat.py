def elastic_critical_load_analysis_frame_3D(node_coords: np.ndarray, elements: Sequence[dict], boundary_conditions: dict[int, Sequence[int]], nodal_loads: dict[int, Sequence[float]]):
    n_nodes = node_coords.shape[0]
    bc_dict = {}
    for (node_idx, bc_spec) in boundary_conditions.items():
        if all((isinstance(x, bool) for x in bc_spec)):
            bc_dict[node_idx] = bc_spec
        else:
            bool_bc = [False] * 6
            for dof_idx in bc_spec:
                if 0 <= dof_idx < 6:
                    bool_bc[dof_idx] = True
            bc_dict[node_idx] = bool_bc
    K_global = assemble_global_stiffness_matrix_linear_elastic_3D(node_coords, elements)
    P_global = assemble_global_load_vector_linear_elastic_3D(nodal_loads, n_nodes)
    (fixed, free) = partition_degrees_of_freedom(bc_dict, n_nodes)
    (u_global, _) = linear_solve(P_global, K_global, fixed, free)
    K_g_global = assemble_global_geometric_stiffness_3D_beam(node_coords, elements, u_global)
    (elastic_critical_load_factor, deformed_shape_vector) = eigenvalue_analysis(K_global, K_g_global, bc_dict, n_nodes)
    return (elastic_critical_load_factor, deformed_shape_vector)