def assemble_global_geometric_stiffness_3D_beam(node_coords: np.ndarray, elements: Sequence[dict], u_global: np.ndarray) -> np.ndarray:
    n_nodes = node_coords.shape[0]
    K = np.zeros((6 * n_nodes, 6 * n_nodes))
    for ele in elements:
        node_i = ele['node_i']
        node_j = ele['node_j']
        (xi, yi, zi) = node_coords[node_i]
        (xj, yj, zj) = node_coords[node_j]
        u_e_global = np.concatenate((u_global[6 * node_i:6 * node_i + 6], u_global[6 * node_j:6 * node_j + 6]))
        local_loads = compute_local_element_loads_beam_3D(ele, xi, yi, zi, xj, yj, zj, u_e_global)
        L = np.linalg.norm([xj - xi, yj - yi, zj - zi])
        Fx2 = local_loads[6]
        Mx2 = local_loads[9]
        My1 = local_loads[4]
        Mz1 = local_loads[5]
        My2 = local_loads[10]
        Mz2 = local_loads[11]
        k_g_local = local_geometric_stiffness_matrix_3D_beam(L, ele['A'], ele['I_rho'], Fx2, Mx2, My1, Mz1, My2, Mz2)
        Gamma = beam_transformation_matrix_3D(xi, yi, zi, xj, yj, zj, ele.get('local_z'))
        k_g_global = Gamma.T @ k_g_local @ Gamma
        dof_indices = np.concatenate((np.arange(6 * node_i, 6 * node_i + 6), np.arange(6 * node_j, 6 * node_j + 6)))
        for a in range(12):
            for b in range(12):
                K[dof_indices[a], dof_indices[b]] += k_g_global[a, b]
    return K