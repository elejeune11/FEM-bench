def elastic_critical_load_analysis_frame_3D(node_coords: np.ndarray, elements: Sequence[dict], boundary_conditions: dict[int, Sequence[int]], nodal_loads: dict[int, Sequence[float]]):
    node_coords = np.asarray(node_coords, dtype=float)
    n_nodes = node_coords.shape[0]
    bc_bool = {}
    for (node_idx, bc_spec) in boundary_conditions.items():
        if len(bc_spec) == 6 and all((isinstance(x, (bool, np.bool_)) for x in bc_spec)):
            bc_bool[node_idx] = bc_spec
        else:
            flags = [False] * 6
            for dof_idx in bc_spec:
                flags[dof_idx] = True
            bc_bool[node_idx] = flags
    elements_processed = []
    for ele in elements:
        ele_dict = dict(ele)
        if 'I_y' in ele_dict:
            ele_dict['I_y'] = ele_dict['I_y']
        elif 'Iy' in ele_dict:
            ele_dict['I_y'] = ele_dict['Iy']
        if 'I_z' in ele_dict:
            ele_dict['I_z'] = ele_dict['I_z']
        elif 'Iz' in ele_dict:
            ele_dict['I_z'] = ele_dict['Iz']
        elements_processed.append(ele_dict)
    K_e_global = assemble_global_stiffness_matrix_linear_elastic_3D(node_coords, elements_processed)
    P_global = assemble_global_load_vector_linear_elastic_3D(nodal_loads, n_nodes)
    (fixed, free) = partition_degrees_of_freedom(bc_bool, n_nodes)
    (u_global, _) = linear_solve(P_global, K_e_global, fixed, free)
    K_g_global = assemble_global_geometric_stiffness_3D_beam(node_coords, elements, u_global)
    (elastic_critical_load_factor, deformed_shape_vector) = eigenvalue_analysis(K_e_global, K_g_global, bc_bool, n_nodes)
    return (elastic_critical_load_factor, deformed_shape_vector)