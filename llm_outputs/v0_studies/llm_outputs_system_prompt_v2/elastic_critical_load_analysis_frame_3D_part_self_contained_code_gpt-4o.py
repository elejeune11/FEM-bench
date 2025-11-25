def elastic_critical_load_analysis_frame_3D_part_self_contained(node_coords: np.ndarray, elements: Sequence[dict], boundary_conditions: dict[int, Sequence[int | bool]], nodal_loads: dict[int, Sequence[float]]):
    n_nodes = node_coords.shape[0]
    n_dofs = 6 * n_nodes
    K_global = np.zeros((n_dofs, n_dofs))
    P_global = np.zeros(n_dofs)
    for element in elements:
        node_i = element['node_i']
        node_j = element['node_j']
        E = element['E']
        nu = element['nu']
        A = element['A']
        Iy = element['Iy']
        Iz = element['Iz']
        J = element['J']
        I_rho = element['I_rho']
        local_z = element['local_z']
        coord_i = node_coords[node_i]
        coord_j = node_coords[node_j]
        L = np.linalg.norm(coord_j - coord_i)
        k_local = local_elastic_stiffness_matrix_3D_beam(E, nu, A, L, Iy, Iz, J)
        T = np.eye(12)
        k_global = T.T @ k_local @ T
        dof_map = np.array([6 * node_i, 6 * node_i + 1, 6 * node_i + 2, 6 * node_i + 3, 6 * node_i + 4, 6 * node_i + 5, 6 * node_j, 6 * node_j + 1, 6 * node_j + 2, 6 * node_j + 3, 6 * node_j + 4, 6 * node_j + 5])
        for a in range(12):
            for b in range(12):
                K_global[dof_map[a], dof_map[b]] += k_global[a, b]
    for (node, load) in nodal_loads.items():
        for i in range(6):
            P_global[6 * node + i] += load[i]
    free_dofs = np.arange(n_dofs)
    for (node, bc) in boundary_conditions.items():
        if isinstance(bc, Sequence) and all((isinstance(x, bool) for x in bc)):
            constrained_dofs = [6 * node + i for (i, fixed) in enumerate(bc) if fixed]
        elif isinstance(bc, Sequence) and all((isinstance(x, int) for x in bc)):
            constrained_dofs = [6 * node + i for i in bc]
        else:
            raise ValueError('Invalid boundary condition format.')
        free_dofs = np.setdiff1d(free_dofs, constrained_dofs)
    K_ff = K_global[np.ix_(free_dofs, free_dofs)]
    P_f = P_global[free_dofs]
    u_f = np.linalg.solve(K_ff, P_f)
    u = np.zeros(n_dofs)
    u[free_dofs] = u_f
    K_g_global = np.zeros((n_dofs, n_dofs))
    for element in elements:
        node_i = element['node_i']
        node_j = element['node_j']
        A = element['A']
        I_rho = element['I_rho']
        coord_i = node_coords[node_i]
        coord_j = node_coords[node_j]
        L = np.linalg.norm(coord_j - coord_i)
        dof_map = np.array([6 * node_i, 6 * node_i + 1, 6 * node_i + 2, 6 * node_i + 3, 6 * node_i + 4, 6 * node_i + 5, 6 * node_j, 6 * node_j + 1, 6 * node_j + 2, 6 * node_j + 3, 6 * node_j + 4, 6 * node_j + 5])
        Fx2 = P_global[dof_map[6]]
        Mx2 = P_global[dof_map[9]]
        My1 = P_global[dof_map[4]]
        Mz1 = P_global[dof_map[5]]
        My2 = P_global[dof_map[10]]
        Mz2 = P_global[dof_map[11]]
        k_g_local = local_geometric_stiffness_matrix_3D_beam(L, A, I_rho, Fx2, Mx2, My1, Mz1, My2, Mz2)
        k_g_global = T.T @ k_g_local @ T
        for a in range(12):
            for b in range(12):
                K_g_global[dof_map[a], dof_map[b]] += k_g_global[a, b]
    K_g_ff = K_g_global[np.ix_(free_dofs, free_dofs)]
    (eigenvalues, eigenvectors) = scipy.linalg.eig(K_ff, -K_g_ff)
    positive_eigenvalues = eigenvalues[eigenvalues > 0]
    if len(positive_eigenvalues) == 0:
        raise ValueError('No positive eigenvalue found.')
    min_eigenvalue_index = np.argmin(positive_eigenvalues)
    elastic_critical_load_factor = positive_eigenvalues[min_eigenvalue_index]
    deformed_shape_vector = np.zeros(n_dofs)
    deformed_shape_vector[free_dofs] = eigenvectors[:, min_eigenvalue_index].real
    return (elastic_critical_load_factor, deformed_shape_vector)