def elastic_critical_load_analysis_frame_3D(node_coords: np.ndarray, elements: Sequence[dict], boundary_conditions: dict[int, Sequence[int]], nodal_loads: dict[int, Sequence[float]]):
    n_nodes = node_coords.shape[0]
    bc_processed = {}
    for (node_idx, dof_spec) in boundary_conditions.items():
        if len(dof_spec) == 6 and all((isinstance(x, (bool, np.bool_)) for x in dof_spec)):
            bc_processed[node_idx] = dof_spec
        else:
            flags = [False] * 6
            for dof_idx in dof_spec:
                if 0 <= dof_idx < 6:
                    flags[dof_idx] = True
            bc_processed[node_idx] = flags
    K_elastic = assemble_global_stiffness_matrix_linear_elastic_3D(node_coords, elements)
    P_ref = assemble_global_load_vector_linear_elastic_3D(nodal_loads, n_nodes)
    (fixed, free) = partition_degrees_of_freedom(bc_processed, n_nodes)
    (u_global, _) = linear_solve(P_ref, K_elastic, fixed, free)
    K_geometric = assemble_global_geometric_stiffness_3D_beam(node_coords, elements, u_global)
    (elastic_critical_load_factor, deformed_shape_vector) = eigenvalue_analysis(K_elastic, K_geometric, bc_processed, n_nodes)
    return (elastic_critical_load_factor, deformed_shape_vector)