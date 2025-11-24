def assemble_global_geometric_stiffness_3D_beam(node_coords: np.ndarray, elements: Sequence[dict], u_global: np.ndarray) -> np.ndarray:
    n_nodes = node_coords.shape[0]
    K = np.zeros((6 * n_nodes, 6 * n_nodes))
    for element in elements:
        i = element['node_i']
        j = element['node_j']
        (xi, yi, zi) = node_coords[i]
        (xj, yj, zj) = node_coords[j]
        u_e_global = np.concatenate([u_global[6 * i:6 * i + 6], u_global[6 * j:6 * j + 6]])
        local_loads = compute_local_element_loads_beam_3D(element, xi, yi, zi, xj, yj, zj, u_e_global)
        Fx2 = local_loads[6]
        Mx2 = local_loads[9]
        My1 = local_loads[4]
        Mz1 = local_loads[5]
        My2 = local_loads[10]
        Mz2 = local_loads[11]
        L = np.linalg.norm([xj - xi, yj - yi, zj - zi])
        A = element['A']
        I_rho = element['I_rho']
        k_g_local = local_geometric_stiffness_matrix_3D_beam(L, A, I_rho, Fx2, Mx2, My1, Mz1, My2, Mz2)
        Gamma = beam_transformation_matrix_3D(xi, yi, zi, xj, yj, zj, element.get('local_z'))
        k_g_global = Gamma.T @ k_g_local @ Gamma
        dofs_i = range(6 * i, 6 * i + 6)
        dofs_j = range(6 * j, 6 * j + 6)
        dofs = np.concatenate([dofs_i, dofs_j])
        for (idx_row, row) in enumerate(dofs):
            for (idx_col, col) in enumerate(dofs):
                K[row, col] += k_g_global[idx_row, idx_col]
    return K