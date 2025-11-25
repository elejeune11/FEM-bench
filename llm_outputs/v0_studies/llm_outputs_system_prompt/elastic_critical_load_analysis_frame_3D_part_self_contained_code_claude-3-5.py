def elastic_critical_load_analysis_frame_3D_part_self_contained(node_coords: np.ndarray, elements: Sequence[dict], boundary_conditions: dict[int, Sequence[int | bool]], nodal_loads: dict[int, Sequence[float]]):
    n_nodes = len(node_coords)
    n_dof = 6 * n_nodes
    K = np.zeros((n_dof, n_dof))
    K_g = np.zeros((n_dof, n_dof))
    P = np.zeros(n_dof)
    for (node, loads) in nodal_loads.items():
        P[6 * node:6 * node + 6] = loads
    for elem in elements:
        (node_i, node_j) = (elem['node_i'], elem['node_j'])
        (xi, yi, zi) = node_coords[node_i]
        (xj, yj, zj) = node_coords[node_j]
        (dx, dy, dz) = (xj - xi, yj - yi, zj - zi)
        L = np.sqrt(dx * dx + dy * dy + dz * dz)
        (cx, cy, cz) = (dx / L, dy / L, dz / L)
        if elem.get('local_z') is not None:
            local_z = np.array(elem['local_z'])
            local_z = local_z / np.linalg.norm(local_z)
        elif abs(cz) < 0.999:
            local_z = np.array([0.0, 0.0, 1.0])
        else:
            local_z = np.array([0.0, 1.0, 0.0])
        local_y = np.cross(local_z, [cx, cy, cz])
        local_y = local_y / np.linalg.norm(local_y)
        local_z = np.cross([cx, cy, cz], local_y)
        T = np.zeros((12, 12))
        R = np.vstack(([cx, cy, cz], local_y, local_z))
        for i in range(4):
            T[3 * i:3 * i + 3, 3 * i:3 * i + 3] = R
        k_local = local_elastic_stiffness_matrix_3D_beam(elem['E'], elem['nu'], elem['A'], L, elem['Iy'], elem['Iz'], elem['J'])
        idx = np.array([6 * node_i + j for j in range(6)] + [6 * node_j + j for j in range(6)])
        k_global = T.T @ k_local @ T
        K[np.ix_(idx, idx)] += k_global
    free_dofs = list(range(n_dof))
    for (node, bc) in boundary_conditions.items():
        if isinstance(bc[0], bool):
            fixed_dofs = [6 * node + i for (i, fixed) in enumerate(bc) if fixed]
        else:
            fixed_dofs = [6 * node + i for i in bc]
        for dof in fixed_dofs:
            if dof in free_dofs:
                free_dofs.remove(dof)
    K_free = K[np.ix_(free_dofs, free_dofs)]
    P_free = P[free_dofs]
    u_free = np.linalg.solve(K_free, P_free)
    u = np.zeros(n_dof)
    u[free_dofs] = u_free
    for elem in elements:
        (node_i, node_j) = (elem['node_i'], elem['node_j'])
        (xi, yi, zi) = node_coords[node_i]
        (xj, yj, zj) = node_coords[node_j]
        (dx, dy, dz) = (xj - xi, yj - yi, zj - zi)
        L = np.sqrt(dx * dx + dy * dy + dz * dz)
        (cx, cy, cz) = (dx / L, dy / L, dz / L)
        if elem.get('local_z') is not None:
            local_z = np.array(elem['local_z'])
            local_z = local_z / np.linalg.norm(local_z)
        elif abs(cz) < 0.999:
            local_z = np.array([0.0, 0.0, 1.0])
        else:
            local_z = np.array([0.0, 1.0, 0.0])
        local_y = np.cross(local_z, [cx, cy, cz])
        local_y = local_y / np.linalg.norm(local_y)
        local_z = np.cross([cx, cy, cz], local_y)
        T = np.zeros((12, 12))
        R = np.vstack(([cx, cy, cz], local_y, local_z))
        for i in range(4):
            T[3 * i:3 * i + 3, 3 * i:3 * i + 3] = R
        idx = np.array([6 * node_i + j for j in range(6)] + [6 * node_j + j for j in range(6)])
        u_elem = u[idx]
        u_local = T @ u_elem
        f_local = k_local @ u_local
        Fx2 = f_local[6]
        Mx2 = f_local[9]
        (My1, Mz1) = (f_local[4], f_local[5])
        (My2, Mz2) = (f_local[10], f_local[11])
        kg_local = local_geometric_stiffness_matrix_3D_beam(L, elem['A'], elem['I_rho'], Fx2, Mx2, My1, Mz1, My2, Mz2)
        kg_global = T.T @ kg_local @ T
        K_g[np.ix_(idx, idx)] += kg_global
    K_free = K[np.ix_(free_dofs, free_dofs)]
    Kg_free = K_g[np.ix_(free_dofs, free_dofs)]
    (eigenvals, eigenvecs) = scipy.linalg.eigh(K_free, -Kg_free)
    pos_eigenvals = eigenvals[eigenvals > 0]
    if len(pos_eigenvals) == 0:
        raise ValueError('No positive eigenvalues found')
    critical_load_factor = np.min(pos_eigenvals)
    mode_index = np.where(eigenvals == critical_load_factor)[0][0]
    mode_shape = np.zeros(n_dof)
    mode_shape[free_dofs] = eigenvecs[:, mode_index]
    return (critical_load_factor, mode_shape)