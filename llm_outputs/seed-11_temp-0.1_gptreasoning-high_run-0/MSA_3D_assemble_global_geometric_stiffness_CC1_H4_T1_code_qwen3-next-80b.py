def MSA_3D_assemble_global_geometric_stiffness_CC1_H4_T1(node_coords: np.ndarray, elements: Sequence[dict], u_global: np.ndarray) -> np.ndarray:
    n_nodes = node_coords.shape[0]
    K_global = np.zeros((6 * n_nodes, 6 * n_nodes))
    for elem in elements:
        i = elem['node_i']
        j = elem['node_j']
        (xi, yi, zi) = node_coords[i]
        (xj, yj, zj) = node_coords[j]
        u_e_global = np.concatenate([u_global[6 * i:6 * i + 6], u_global[6 * j:6 * j + 6]])
        load_local = compute_local_element_loads_beam_3D(elem, xi, yi, zi, xj, yj, zj, u_e_global)
        (Fxi, Fyi, Fzi, Mxi, Myi, Mzi, Fxj, Fyj, Fzj, Mxj, Myj, Mzj) = load_local
        L = np.linalg.norm([xj - xi, yj - yi, zj - zi])
        Fx2 = Fxj
        Mx2 = Mxj
        My1 = Myi
        Mz1 = Mzi
        My2 = Myj
        Mz2 = Mzj
        k_g_local = local_geometric_stiffness_matrix_3D_beam(L, elem['A'], elem['I_rho'], Fx2, Mx2, My1, Mz1, My2, Mz2)
        Gamma = beam_transformation_matrix_3D(xi, yi, zi, xj, yj, zj, elem.get('local_z'))
        k_g_global = Gamma.T @ k_g_local @ Gamma
        dofs = np.concatenate([np.arange(6 * i, 6 * i + 6), np.arange(6 * j, 6 * j + 6)])
        for (idx1, global_idx1) in enumerate(dofs):
            for (idx2, global_idx2) in enumerate(dofs):
                K_global[global_idx1, global_idx2] += k_g_global[idx1, idx2]
    return K_global