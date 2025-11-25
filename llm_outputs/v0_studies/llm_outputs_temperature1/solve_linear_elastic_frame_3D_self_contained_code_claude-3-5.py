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
        R = np.column_stack((x_local, y_local, z_local))
        for i in range(4):
            T[3 * i:3 * i + 3, 3 * i:3 * i + 3] = R
        G = E / (2 * (1 + nu))
        ke = np.zeros((12, 12))
        ke[0, 0] = ke[6, 6] = E * A / L
        ke[0, 6] = ke[6, 0] = -E * A / L
        ke[3, 3] = ke[9, 9] = G * J / L
        ke[3, 9] = ke[9, 3] = -G * J / L
        ke[1, 1] = ke[7, 7] = 12 * E * Iz / L ** 3
        ke[1, 7] = ke[7, 1] = -12 * E * Iz / L ** 3
        ke[5, 5] = ke[11, 11] = 4 * E * Iz / L
        ke[5, 11] = ke[11, 5] = 2 * E * Iz / L
        ke[1, 5] = ke[5, 1] = 6 * E * Iz / L ** 2
        ke[1, 11] = ke[11, 1] = -6 * E * Iz / L ** 2
        ke[7, 5] = ke[5, 7] = -6 * E * Iz / L ** 2
        ke[7, 11] = ke[11, 7] = 6 * E * Iz / L ** 2
        ke[2, 2] = ke[8, 8] = 12 * E * Iy / L ** 3
        ke[2, 8] = ke[8, 2] = -12 * E * Iy / L ** 3
        ke[4, 4] = ke[10, 10] = 4 * E * Iy / L
        ke[4, 10] = ke[10, 4] = 2 * E * Iy / L
        ke[2, 4] = ke[4, 2] = -6 * E * Iy / L ** 2
        ke[2, 10] = ke[10, 2] = 6 * E * Iy / L ** 2
        ke[8, 4] = ke[4, 8] = 6 * E * Iy / L ** 2
        ke[8, 10] = ke[10, 8] = -6 * E * Iy / L ** 2
        ke_global = T.T @ ke @ T
        dofs_i = np.array([6 * node_i + i for i in range(6)])
        dofs_j = np.array([6 * node_j + i for i in range(6)])
        dofs = np.concatenate((dofs_i, dofs_j))
        for i in range(12):
            for j in range(12):
                K[dofs[i], dofs[j]] += ke_global[i, j]
    free_dofs = []
    fixed_dofs = []
    for i in range(n_nodes):
        for j in range(6):
            dof = 6 * i + j
            if i in boundary_conditions and boundary_conditions[i][j]:
                fixed_dofs.append(dof)
            else:
                free_dofs.append(dof)
    free_dofs = np.array(free_dofs)
    fixed_dofs = np.array(fixed_dofs)
    Kff = K[np.ix_(free_dofs, free_dofs)]
    Kfp = K[np.ix_(free_dofs, fixed_dofs)]
    Pf = P[free_dofs]
    if np.linalg.cond(Kff) >= 1e+16:
        raise ValueError('System is ill-conditioned')
    uf = np.linalg.solve(Kff, Pf)
    u = np.zeros(n_dofs)
    u[free_dofs] = uf
    r = np.zeros(n_dofs)
    r[fixed_dofs] = K[fixed_dofs, :] @ u - P[fixed_dofs]
    return (u, r)