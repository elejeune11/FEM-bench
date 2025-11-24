def elastic_critical_load_analysis_frame_3D(node_coords: np.ndarray, elements: Sequence[dict], boundary_conditions: dict[int, Sequence[int]], nodal_loads: dict[int, Sequence[float]]):
    node_coords = np.asarray(node_coords, dtype=float)
    n_nodes = node_coords.shape[0]
    elements_norm = []
    for ele in elements:
        e = dict(ele)
        if 'I_y' not in e and 'Iy' in e:
            e['I_y'] = e['Iy']
        if 'I_z' not in e and 'Iz' in e:
            e['I_z'] = e['Iz']
        elements_norm.append(e)
    bc_bool = {}
    for (n, spec) in (boundary_conditions or {}).items():
        arr = list(spec) if spec is not None else []
        if len(arr) == 6:
            mask = np.asarray(arr, dtype=bool)
        else:
            mask = np.zeros(6, dtype=bool)
            idxs = np.asarray(arr, dtype=int).ravel()
            for i in idxs:
                if 0 <= int(i) < 6:
                    mask[int(i)] = True
        bc_bool[int(n)] = mask
    K_e = assemble_global_stiffness_matrix_linear_elastic_3D(node_coords, elements_norm)
    P = assemble_global_load_vector_linear_elastic_3D(nodal_loads or {}, n_nodes)
    (fixed, free) = partition_degrees_of_freedom(bc_bool, n_nodes)
    (u_ref, _) = linear_solve(P, K_e, fixed, free)
    K_g = assemble_global_geometric_stiffness_3D_beam(node_coords, elements_norm, u_ref)
    (lam, mode) = eigenvalue_analysis(K_e, K_g, bc_bool, n_nodes)
    return (lam, mode)