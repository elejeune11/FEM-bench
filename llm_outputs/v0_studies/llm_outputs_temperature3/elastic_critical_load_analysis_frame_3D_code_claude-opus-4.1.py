def elastic_critical_load_analysis_frame_3D(node_coords: np.ndarray, elements: Sequence[dict], boundary_conditions: dict[int, Sequence[int]], nodal_loads: dict[int, Sequence[float]]):
    n_nodes = node_coords.shape[0]
    bc_bool = {}
    for (node_idx, bc) in boundary_conditions.items():
        if len(bc) == 6 and all((isinstance(b, bool) for b in bc)):
            bc_bool[node_idx] = bc
        else:
            flags = [False] * 6
            for dof_idx in bc:
                flags[dof_idx] = True
            bc_bool[node_idx] = flags
    K_e = assemble_global_stiffness_matrix_linear_elastic_3D(node_coords, elements)
    P = assemble_global_load_vector_linear_elastic_3D(nodal_loads, n_nodes)
    (fixed, free) = partition_degrees_of_freedom(bc_bool, n_nodes)
    (u, _) = linear_solve(P, K_e, fixed, free)
    K_g = assemble_global_geometric_stiffness_3D_beam(node_coords, elements, u)
    (elastic_critical_load_factor, deformed_shape_vector) = eigenvalue_analysis(K_e, K_g, bc_bool, n_nodes)
    return (elastic_critical_load_factor, deformed_shape_vector)