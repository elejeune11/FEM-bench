def elastic_critical_load_analysis_frame_3D(node_coords: np.ndarray, elements: Sequence[dict], boundary_conditions: dict[int, Sequence[int]], nodal_loads: dict[int, Sequence[float]]):
    n_nodes = node_coords.shape[0]
    bc_bool = {}
    for (node_idx, bc) in boundary_conditions.items():
        if len(bc) == 6 and all((isinstance(b, (bool, np.bool_)) for b in bc)):
            bc_bool[node_idx] = bc
        else:
            flags = [False] * 6
            for dof_idx in bc:
                flags[dof_idx] = True
            bc_bool[node_idx] = flags
    elements_fixed = []
    for ele in elements:
        ele_copy = dict(ele)
        if 'Iy' in ele_copy and 'I_y' not in ele_copy:
            ele_copy['I_y'] = ele_copy['Iy']
        if 'Iz' in ele_copy and 'I_z' not in ele_copy:
            ele_copy['I_z'] = ele_copy['Iz']
        elements_fixed.append(ele_copy)
    K_e_global = assemble_global_stiffness_matrix_linear_elastic_3D(node_coords, elements_fixed)
    P_global = assemble_global_load_vector_linear_elastic_3D(nodal_loads, n_nodes)
    (fixed, free) = partition_degrees_of_freedom(bc_bool, n_nodes)
    (u_global, _) = linear_solve(P_global, K_e_global, fixed, free)
    K_g_global = assemble_global_geometric_stiffness_3D_beam(node_coords, elements_fixed, u_global)
    (elastic_critical_load_factor, deformed_shape_vector) = eigenvalue_analysis(K_e_global, K_g_global, bc_bool, n_nodes)
    return (elastic_critical_load_factor, deformed_shape_vector)