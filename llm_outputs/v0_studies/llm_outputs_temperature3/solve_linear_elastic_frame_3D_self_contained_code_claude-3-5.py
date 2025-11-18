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
        else:
            z_ref = np.array([0, 0, 1])
        x_local = dx
        y_local = np.cross(z_ref, x_local)
        y_local = y_local / np.linalg.norm(y_local)
        z_local = np.cross(x_local, y_local)
        T = np.zeros((12, 12))
        R = np.vstack((x_local, y_local, z_local))
        for i in range(4):
            T[3 * i:3 * i + 3, 3 * i:3 * i + 3] = R
        G = E / (2 * (1 + nu))
        ke = np.zeros((12, 12))
        ke[0, 0] = ke[6, 6] = E * A / L
        ke[0, 6] = ke[6, 0] = -E * A / L
        ke[3, 3] = ke[9, 9] = G * J / L
        ke[3, 9] = ke[9, 3] = -G * J / L
        for (axis, I) in [(1, Iz), (2, Iy)]:
            (i1, i2) = (axis, axis + 3)
            (j1, j2) = (axis + 6, axis + 9)
            ke[i1, i1] = ke[j1, j1] = 12 * E * I / L ** 3
            ke[i2, i2] = ke[j2, j2] = 4 * E * I / L
            ke[i1, i2] = ke[i2, i1] = 6 * E * I / L ** 2
            ke[i1, j1] = ke[j1, i1] = -12 * E * I / L ** 3
            ke[i1, j2] = ke[j2, i1] = 6 * E * I / L ** 2
            ke[i2, j1] = ke[j1, i2] = -6 * E * I / L ** 2
            ke[i2, j2] = ke[j2, i2] = 2 * E * I / L
            ke[j1, j2] = ke[j2, j1] = -6 * E * I / L ** 2
        ke = T.T @ ke @ T
        dofs_i = slice(6 * node_i, 6 * node_i + 6)
        dofs_j = slice(6 * node_j, 6 * node_j + 6)
        K[np.ix_([*range(6 * node_i, 6 * node_i + 6), *range(6 * node_j, 6 * node_j + 6)], [*range(6 * node_i, 6 * node_i + 6), *range(6 * node_j, 6 * node_j + 6)])] += ke
    fixed_dofs = []
    for (node, fixity) in boundary_conditions.items():
        fixed_dofs.extend([6 * node + i for (i, fix) in enumerate(fixity) if fix])
    free_dofs = list(set(range(n_dofs)) - set(fixed_dofs))
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