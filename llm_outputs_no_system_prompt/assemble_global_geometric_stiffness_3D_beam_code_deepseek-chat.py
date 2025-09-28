n_nodes = node_coords.shape[0]
    n_dofs = 6 * n_nodes
    K_g_global = np.zeros((n_dofs, n_dofs))
    for ele in elements:
        i = ele['node_i']
        j = ele['node_j']
        xi, yi, zi = node_coords[i]
        xj, yj, zj = node_coords[j]
        ele_dofs = [6*i, 6*i+1, 6*i+2, 6*i+3, 6*i+4, 6*i+5,
                    6*j, 6*j+1, 6*j+2, 6*j+3, 6*j+4, 6*j+5]
        u_ele_global = u_global[ele_dofs]
        Gamma = beam_transformation_matrix_3D(xi, yi, zi, xj, yj, zj, ele.get('local_z'))
        local_loads = compute_local_element_loads_beam_3D(ele, xi, yi, zi, xj, yj, zj, u_ele_global)
        Fx2 = local_loads[6]
        Mx2 = local_loads[9]
        My1 = local_loads[4]
        Mz1 = local_loads[5]
        My2 = local_loads[10]
        Mz2 = local_loads[11]
        L = np.linalg.norm([xj - xi, yj - yi, zj - zi])
        k_g_local = local_geometric_stiffness_matrix_3D_beam(
            L, ele['A'], ele['I_rho'], Fx2, Mx2, My1, Mz1, My2, Mz2
        )
        k_g_global_ele = Gamma.T @ k_g_local @ Gamma
        for idx_row, dof_row in enumerate(ele_dofs):
            for idx_col, dof_col in enumerate(ele_dofs):
                K_g_global[dof_row, dof_col] += k_g_global_ele[idx_row, idx_col]
    return K_g_global