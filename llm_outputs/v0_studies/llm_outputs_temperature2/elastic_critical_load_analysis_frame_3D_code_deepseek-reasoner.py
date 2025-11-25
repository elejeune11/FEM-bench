def elastic_critical_load_analysis_frame_3D(node_coords: np.ndarray, elements: Sequence[dict], boundary_conditions: dict[int, Sequence[int]], nodal_loads: dict[int, Sequence[float]]):
    n_nodes = node_coords.shape[0]
    bc_bool = {}
    for (node, spec) in boundary_conditions.items():
        if len(spec) == 6 and all((isinstance(x, (bool, np.bool_)) for x in spec)):
            bc_bool[node] = spec
        else:
            flags = [False] * 6
            for dof_idx in spec:
                if 0 <= dof_idx < 6:
                    flags[dof_idx] = True
            bc_bool[node] = flags
    K_e_global = assemble_global_stiffness_matrix_linear_elastic_3D(node_coords, elements)
    P_global = assemble_global_load_vector_linear_elastic_3D(nodal_loads, n_nodes)
    (fixed, free) = partition_degrees_of_freedom(bc_bool, n_nodes)
    (u_global, _) = linear_solve(P_global, K_e_global, fixed, free)
    K_g_global = assemble_global_geometric_stiffness_3D_beam(node_coords, elements, u_global)
    (elastic_critical_load_factor, deformed_shape_vector) = eigenvalue_analysis(K_e_global, K_g_global, bc_bool, n_nodes)
    return (elastic_critical_load_factor, deformed_shape_vector)