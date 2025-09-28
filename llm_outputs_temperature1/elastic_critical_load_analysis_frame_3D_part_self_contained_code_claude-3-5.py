def elastic_critical_load_analysis_frame_3D_part_self_contained(node_coords: np.ndarray, elements: Sequence[dict], boundary_conditions: dict[int, Sequence[int | bool]], nodal_loads: dict[int, Sequence[float]]):
    n_nodes = len(node_coords)
    n_dof = 6 * n_nodes
    K = np.zeros((n_dof, n_dof))
    K_g = np.zeros((n_dof, n_dof))
    P = np.zeros(n_dof)
    for (node_idx, loads) in nodal_loads.items():
        P[6 * node_idx:6 * node_idx + 6] = loads
    constrained_dofs = set()
    for (node_idx, bc_spec) in boundary_conditions.items():
        if isinstance(bc_spec[0], bool):
            constrained_dofs.update((6 * node_idx + i for (i, fixed) in enumerate(bc_spec) if fixed))
        else:
            constrained_dofs.update((6 * node_idx + i for i in bc_spec))
    free_dofs = list(set(range(n_dof)) - constrained_dofs)
    for elem in elements:
        (i, j) = (elem['node_i'], elem['node_j'])
        (xi, yi, zi) = node_coords[i]
        (xj, yj, zj) = node_coords[j]
        (dx, dy, dz) = (xj - xi, yj - yi, zj - zi)
        L = np.sqrt(dx * dx + dy * dy + dz * dz)
        ex = np.array([dx, dy, dz]) / L
        if elem['local_z'] is None:
            if abs(ex[2]) < 0.99:
                ez = np.cross([0, 0, 1], ex)
                ez = ez / np.linalg.norm(ez)
            else:
                ez = np.cross([1, 0, 0], ex)
                ez = ez / np.linalg.norm(ez)
        else:
            ez = np.array(elem['local_z'])
            ez = ez - np.dot(ez, ex) * ex
            ez = ez / np.linalg.norm(ez)
        ey = np.cross(ez, ex)
        T = np.zeros((3, 3))
        T[:, 0] = ex
        T[:, 1] = ey
        T[:, 2] = ez
        Lambda = np.zeros((12, 12))
        for k in range(4):
            Lambda[3 * k:3 * k + 3, 3 * k:3 * k + 3] = T
        k_el = local_elastic_stiffness_matrix_3D_beam(elem['E'], elem['nu'], elem['A'], L, elem['Iy'], elem['Iz'], elem['J'])
        k_el_global = Lambda.T @ k_el @ Lambda
        dofs = [6 * i + d for d in range(6)] + [6 * j + d for d in range(6)]
        for (p, I) in enumerate(dofs):
            for (q, J) in enumerate(dofs):
                K[I, J] += k_el_global[p, q]
    u = np.zeros(n_dof)
    K_ff = K[np.ix_(free_dofs, free_dofs)]
    P_f = P[free_dofs]
    u_f = np.linalg.solve(K_ff, P_f)
    u[free_dofs] = u_f
    for elem in elements:
        (i, j) = (elem['node_i'], elem['node_j'])
        (xi, yi, zi) = node_coords[i]
        (xj, yj, zj) = node_coords[j]
        (dx, dy, dz) = (xj - xi, yj - yi, zj - zi)
        L = np.sqrt(dx * dx + dy * dy + dz * dz)
        ex = np.array([dx, dy, dz]) / L
        if elem['local_z'] is None:
            if abs(ex[2]) < 0.99:
                ez = np.cross([0, 0, 1], ex)
                ez = ez / np.linalg.norm(ez)
            else:
                ez = np.cross([1, 0, 0], ex)
                ez = ez / np.linalg.norm(ez)
        else:
            ez = np.array(elem['local_z'])
            ez = ez - np.dot(ez, ex) * ex
            ez = ez / np.linalg.norm(ez)
        ey = np.cross(ez, ex)
        T = np.zeros((3, 3))
        T[:, 0] = ex
        T[:, 1] = ey
        T[:, 2] = ez
        Lambda = np.zeros((12, 12))
        for k in range(4):
            Lambda[3 * k:3 * k + 3, 3 * k:3 * k + 3] = T
        dofs = [6 * i + d for d in range(6)] + [6 * j + d for d in range(6)]
        u_el = u[dofs]
        u_el_local = Lambda @ u_el
        f_el_local = k_el @ u_el_local
        k_g_el = local_geometric_stiffness_matrix_3D_beam(L, elem['A'], elem['I_rho'], f_el_local[6], f_el_local[9], f_el_local[4], f_el_local[5], f_el_local[10], f_el_local[11])
        k_g_el_global = Lambda.T @ k_g_el @ Lambda
        for (p, I) in enumerate(dofs):
            for (q, J) in enumerate(dofs):
                K_g[I, J] += k_g_el_global[p, q]
    K_ff = K[np.ix_(free_dofs, free_dofs)]
    K_g_ff = K_g[np.ix_(free_dofs, free_dofs)]
    (eigenvals, eigenvecs) = scipy.linalg.eigh(K_ff, -K_g_ff)
    pos_eigenvals = eigenvals[eigenvals > 0]
    if len(pos_eigenvals) == 0:
        raise ValueError('No positive eigenvalues found')
    critical_load_factor = pos_eigenvals[0]
    mode_shape = np.zeros(n_dof)
    mode_shape[free_dofs] = eigenvecs[:, np.where(eigenvals == critical_load_factor)[0][0]]
    return (critical_load_factor, mode_shape)