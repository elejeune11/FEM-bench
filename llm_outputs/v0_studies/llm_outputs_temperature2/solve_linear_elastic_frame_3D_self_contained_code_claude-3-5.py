def solve_linear_elastic_frame_3D_self_contained(node_coords: np.ndarray, elements: Sequence[dict], boundary_conditions: dict[int, Sequence[int]], nodal_loads: dict[int, Sequence[float]]):
    n_nodes = len(node_coords)
    n_dofs = 6 * n_nodes
    K = np.zeros((n_dofs, n_dofs))
    P = np.zeros(n_dofs)
    for (node_idx, loads) in nodal_loads.items():
        dof_start = 6 * node_idx
        P[dof_start:dof_start + 6] = loads
    for elem in elements:
        (node_i, node_j) = (elem['node_i'], elem['node_j'])
        (E, nu) = (elem['E'], elem['nu'])
        (A, Iy, Iz, J) = (elem['A'], elem['I_y'], elem['I_z'], elem['J'])
        G = E / (2 * (1 + nu))
        xi = node_coords[node_i]
        xj = node_coords[node_j]
        L = np.linalg.norm(xj - xi)
        dx = (xj - xi) / L
        if elem.get('local_z') is not None:
            z_ref = elem['local_z']
        elif abs(dx[2]) < 0.99:
            z_ref = np.array([0, 0, 1])
        else:
            z_ref = np.array([0, 1, 0])
        x_local = dx
        y_local = np.cross(z_ref, x_local)
        y_local /= np.linalg.norm(y_local)
        z_local = np.cross(x_local, y_local)
        T = np.zeros((12, 12))
        R = np.vstack((x_local, y_local, z_local))
        for i in range(4):
            T[3 * i:3 * i + 3, 3 * i:3 * i + 3] = R
        k = np.zeros((12, 12))
        k[0, 0] = k[6, 6] = E * A / L
        k[0, 6] = k[6, 0] = -E * A / L
        k[3, 3] = k[9, 9] = G * J / L
        k[3, 9] = k[9, 3] = -G * J / L
        EIy = E * Iy
        EIz = E * Iz
        k[1, 1] = k[7, 7] = 12 * EIz / L ** 3
        k[1, 7] = k[7, 1] = -12 * EIz / L ** 3
        k[1, 5] = k[5, 1] = 6 * EIz / L ** 2
        k[1, 11] = k[11, 1] = 6 * EIz / L ** 2
        k[5, 5] = k[11, 11] = 4 * EIz / L
        k[5, 7] = k[7, 5] = -6 * EIz / L ** 2
        k[5, 11] = k[11, 5] = 2 * EIz / L
        k[7, 11] = k[11, 7] = -6 * EIz / L ** 2
        k[2, 2] = k[8, 8] = 12 * EIy / L ** 3
        k[2, 8] = k[8, 2] = -12 * EIy / L ** 3
        k[2, 4] = k[4, 2] = -6 * EIy / L ** 2
        k[2, 10] = k[10, 2] = -6 * EIy / L ** 2
        k[4, 4] = k[10, 10] = 4 * EIy / L
        k[4, 8] = k[8, 4] = 6 * EIy / L ** 2
        k[4, 10] = k[10, 4] = 2 * EIy / L
        k[8, 10] = k[10, 8] = 6 * EIy / L ** 2
        k_global = T.T @ k @ T
        dofs_i = np.array([6 * node_i + i for i in range(6)])
        dofs_j = np.array([6 * node_j + i for i in range(6)])
        dofs = np.concatenate((dofs_i, dofs_j))
        for i in range(12):
            for j in range(12):
                K[dofs[i], dofs[j]] += k_global[i, j]
    free_dofs = []
    fixed_dofs = []
    for i in range(n_nodes):
        bc = boundary_conditions.get(i, [0] * 6)
        for j in range(6):
            dof = 6 * i + j
            if bc[j] == 0:
                free_dofs.append(dof)
            else:
                fixed_dofs.append(dof)
    free_dofs = np.array(free_dofs)
    fixed_dofs = np.array(fixed_dofs)
    Kff = K[np.ix_(free_dofs, free_dofs)]
    Kfp = K[np.ix_(free_dofs, fixed_dofs)]
    Pf = P[free_dofs]
    if np.linalg.cond(Kff) >= 1e+16:
        raise ValueError('Free-free stiffness matrix is ill-conditioned')
    uf = np.linalg.solve(Kff, Pf)
    u = np.zeros(n_dofs)
    u[free_dofs] = uf
    r = K @ u - P
    r[free_dofs] = 0
    return (u, r)