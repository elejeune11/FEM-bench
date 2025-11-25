def assemble_global_geometric_stiffness_3D_beam(node_coords: np.ndarray, elements: Sequence[dict], u_global: np.ndarray) -> np.ndarray:
    n_nodes = len(node_coords)
    K = np.zeros((6 * n_nodes, 6 * n_nodes))
    for ele in elements:
        (i, j) = (ele['node_i'], ele['node_j'])
        (xi, yi, zi) = node_coords[i]
        (xj, yj, zj) = node_coords[j]
        dx = xj - xi
        dy = yj - yi
        dz = zj - zi
        L = np.sqrt(dx * dx + dy * dy + dz * dz)
        Gamma = beam_transformation_matrix_3D(xi, yi, zi, xj, yj, zj, ele.get('local_z'))
        dofs_i = slice(6 * i, 6 * i + 6)
        dofs_j = slice(6 * j, 6 * j + 6)
        u_e_global = np.concatenate([u_global[dofs_i], u_global[dofs_j]])
        f_local = compute_local_element_loads_beam_3D(ele, xi, yi, zi, xj, yj, zj, u_e_global)
        Fx2 = f_local[6]
        Mx2 = f_local[9]
        (My1, Mz1) = (f_local[4], f_local[5])
        (My2, Mz2) = (f_local[10], f_local[11])
        k_g_local = local_geometric_stiffness_matrix_3D_beam(L, ele['A'], ele['I_rho'], Fx2, Mx2, My1, Mz1, My2, Mz2)
        k_g_global = Gamma.T @ k_g_local @ Gamma
        dofs = np.concatenate([np.arange(6 * i, 6 * i + 6), np.arange(6 * j, 6 * j + 6)])
        for (p, dof_p) in enumerate(dofs):
            for (q, dof_q) in enumerate(dofs):
                K[dof_p, dof_q] += k_g_global[p, q]
    return K