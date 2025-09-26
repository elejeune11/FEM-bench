def assemble_global_geometric_stiffness_3D_beam(node_coords: np.ndarray, elements: Sequence[dict], u_global: np.ndarray) -> np.ndarray:
    n_nodes = node_coords.shape[0]
    K_global = np.zeros((6 * n_nodes, 6 * n_nodes))
    for element in elements:
        i = element['node_i']
        j = element['node_j']
        (xi, yi, zi) = node_coords[i]
        (xj, yj, zj) = node_coords[j]
        dof_i = 6 * i
        dof_j = 6 * j
        u_elem = np.concatenate([u_global[dof_i:dof_i + 6], u_global[dof_j:dof_j + 6]])
        local_z = element.get('local_z')
        Gamma = beam_transformation_matrix_3D(xi, yi, zi, xj, yj, zj, local_z)
        local_loads = compute_local_element_loads_beam_3D(element, xi, yi, zi, xj, yj, zj, u_elem)
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
        k_g_global = Gamma.T @ k_g_local @ Gamma
        indices = np.array([dof_i, dof_i + 1, dof_i + 2, dof_i + 3, dof_i + 4, dof_i + 5, dof_j, dof_j + 1, dof_j + 2, dof_j + 3, dof_j + 4, dof_j + 5])
        for row in range(12):
            for col in range(12):
                K_global[indices[row], indices[col]] += k_g_global[row, col]
    return K_global