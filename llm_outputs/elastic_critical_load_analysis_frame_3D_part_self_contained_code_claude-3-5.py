def elastic_critical_load_analysis_frame_3D_part_self_contained(node_coords: np.ndarray, elements: Sequence[dict], boundary_conditions: dict[int, Sequence[int | bool]], nodal_loads: dict[int, Sequence[float]]):
    n_nodes = len(node_coords)
    n_dof = 6 * n_nodes
    K = np.zeros((n_dof, n_dof))
    K_g = np.zeros((n_dof, n_dof))
    P = np.zeros(n_dof)
    for (node_idx, loads) in nodal_loads.items():
        dofs = slice(6 * node_idx, 6 * node_idx + 6)
        P[dofs] = loads
    constrained_dofs = set()
    for (node_idx, bc_spec) in boundary_conditions.items():
        if isinstance(bc_spec[0], bool):
            constrained_dofs.update((6 * node_idx + i for (i, fixed) in enumerate(bc_spec) if fixed))
        else:
            constrained_dofs.update((6 * node_idx + i for i in bc_spec))
    free_dofs = list(set(range(n_dof)) - constrained_dofs)
    for elem in elements:
        (node_i, node_j) = (elem['node_i'], elem['node_j'])
        (xi, yi, zi) = node_coords[node_i]
        (xj, yj, zj) = node_coords[node_j]
        (dx, dy, dz) = (xj - xi, yj - yi, zj - zi)
        L = np.sqrt(dx * dx + dy * dy + dz * dz)
        (cx, cy, cz) = (dx / L, dy / L, dz / L)
        if elem.get('local_z') is not None:
            z_vec = np.array(elem['local_z'])
            z_vec = z_vec / np.linalg.norm(z_vec)
            y_vec = np.cross(z_vec, [cx, cy, cz])
            y_vec = y_vec / np.linalg.norm(y_vec)
            z_vec = np.cross([cx, cy, cz], y_vec)
        else:
            temp = np.cross([cx, cy, cz], [0, 0, 1])
            if np.allclose(temp, 0):
                temp = np.cross([cx, cy, cz], [0, 1, 0])
            y_vec = temp / np.linalg.norm(temp)
            z_vec = np.cross([cx, cy, cz], y_vec)
        R = np.zeros((3, 3))
        R[:, 0] = [cx, cy, cz]
        R[:, 1] = y_vec
        R[:, 2] = z_vec
        T = np.zeros((12, 12))
        for i in range(4):
            T[3 * i:3 * i + 3, 3 * i:3 * i + 3] = R
        k_local = local_elastic_stiffness_matrix_3D_beam(elem['E'], elem['nu'], elem['A'], L, elem['Iy'], elem['Iz'], elem['J'])
        k_global = T.T @ k_local @ T
        dofs = np.concatenate([np.arange(6 * node_i, 6 * node_i + 6), np.arange(6 * node_j, 6 * node_j + 6)])
        for (i, gi) in enumerate(dofs):
            for (j, gj) in enumerate(dofs):
                K[gi, gj] += k_global[i, j]
    K_ff = K[np.ix_(free_dofs, free_dofs)]
    P_f = P[free_dofs]
    u_f = np.linalg.solve(K_ff, P_f)
    u = np.zeros(n_dof)
    u[free_dofs] = u_f
    for elem in elements:
        (node_i, node_j) = (elem['node_i'], elem['node_j'])
        (xi, yi, zi) = node_coords[node_i]
        (xj, yj, zj) = node_coords[node_j]
        (dx, dy, dz) = (xj - xi, yj - yi, zj - zi)
        L = np.sqrt(dx * dx + dy * dy + dz * dz)
        (cx, cy, cz) = (dx / L, dy / L, dz / L)
        if elem.get('local_z') is not None:
            z_vec = np.array(elem['local_z'])
            z_vec = z_vec / np.linalg.norm(z_vec)
            y_vec = np.cross(z_vec, [cx, cy, cz])
            y_vec = y_vec / np.linalg.norm(y_vec)
            z_vec = np.cross([cx, cy, cz], y_vec)
        else:
            temp = np.cross([cx, cy, cz], [0, 0, 1])
            if np.allclose(temp, 0):
                temp = np.cross([cx, cy, cz], [0, 1, 0])
            y_vec = temp / np.linalg.norm(temp)
            z_vec = np.cross([cx, cy, cz], y_vec)
        R = np.zeros((3, 3))
        R[:, 0] = [cx, cy, cz]
        R[:, 1] = y_vec
        R[:, 2] = z_vec
        T = np.zeros((12, 12))
        for i in range(4):
            T[3 * i:3 * i + 3, 3 * i:3 * i + 3] = R
        dofs = np.concatenate([np.arange(6 * node_i, 6 * node_i + 6), np.arange(6 * node_j, 6 * node_j + 6)])
        u_elem = u[dofs]
        f_elem_global = k_global @ u_elem
        f_elem_local = T @ f_elem_global
        Fx2 = f_elem_local[6]
        Mx2 = f_elem_local[9]
        (My1, Mz1) = (f_elem_local[4], f_elem_local[5])
        (My2, Mz2) = (f_elem_local[10], f_elem_local[11])
        kg_local = local_geometric_stiffness_matrix_3D_beam(L, elem['A'], elem['I_rho'], Fx2, Mx2, My1, Mz1, My2, Mz2)
        kg_global = T.T @ kg_local @ T
        for (i, gi) in enumerate(dofs):
            for (j, gj) in enumerate(dofs):
                K_g[gi, gj] += kg_global[i, j]