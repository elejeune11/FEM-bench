def MSA_3D_assemble_global_geometric_stiffness_CC1_H4_T1(node_coords: np.ndarray, elements: Sequence[dict], u_global: np.ndarray) -> np.ndarray:
    n_nodes = node_coords.shape[0]
    n_dofs = 6 * n_nodes
    K_g = np.zeros((n_dofs, n_dofs))
    for ele in elements:
        (node_i, node_j) = (ele['node_i'], ele['node_j'])
        (xi, yi, zi) = node_coords[node_i]
        (xj, yj, zj) = node_coords[node_j]
        L = np.linalg.norm(node_coords[node_j] - node_coords[node_i])
        Gamma = beam_transformation_matrix_3D(xi, yi, zi, xj, yj, zj, ele.get('local_z'))
        u_e_global = u_global[[node_i * 6, node_i * 6 + 1, node_i * 6 + 2, node_i * 6 + 3, node_i * 6 + 4, node_i * 6 + 5, node_j * 6, node_j * 6 + 1, node_j * 6 + 2, node_j * 6 + 3, node_j * 6 + 4, node_j * 6 + 5]]
        load_dofs_local = compute_local_element_loads_beam_3D(ele, xi, yi, zi, xj, yj, zj, u_e_global)
        Fx2 = load_dofs_local[6]
        Mx2 = load_dofs_local[9]
        My1 = load_dofs_local[4]
        Mz1 = load_dofs_local[5]
        My2 = load_dofs_local[10]
        Mz2 = load_dofs_local[11]
        k_g_local = local_geometric_stiffness_matrix_3D_beam(L, ele['A'], ele['I_rho'], Fx2, Mx2, My1, Mz1, My2, Mz2)
        k_g_global = Gamma.T @ k_g_local @ Gamma
        dofs_i = np.arange(node_i * 6, (node_i + 1) * 6)
        dofs_j = np.arange(node_j * 6, (node_j + 1) * 6)
        dofs = np.concatenate([dofs_i, dofs_j])
        K_g[np.ix_(dofs, dofs)] += k_g_global
    return K_g