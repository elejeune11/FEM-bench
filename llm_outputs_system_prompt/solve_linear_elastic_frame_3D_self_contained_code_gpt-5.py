def solve_linear_elastic_frame_3D_self_contained(node_coords: np.ndarray, elements: Sequence[dict], boundary_conditions: dict[int, Sequence[int]], nodal_loads: dict[int, Sequence[float]]):
    import numpy as np
    N = int(node_coords.shape[0])
    ndof_per_node = 6
    ndof = ndof_per_node * N
    K = np.zeros((ndof, ndof), dtype=float)
    P = np.zeros(ndof, dtype=float)

    def node_dofs(n):
        base = n * ndof_per_node
        return np.arange(base, base + ndof_per_node, dtype=int)
    if nodal_loads is not None:
        for (n, loads) in nodal_loads.items():
            dofs = node_dofs(int(n))
            l = np.asarray(loads, dtype=float)
            if l.size != 6:
                raise ValueError
            P[dofs] += l
    for e in elements:
        i = int(e['node_i'])
        j = int(e['node_j'])
        xi = np.asarray(node_coords[i], dtype=float)
        xj = np.asarray(node_coords[j], dtype=float)
        vec = xj - xi
        L = float(np.linalg.norm(vec))
        if not np.isfinite(L) or L <= 0.0:
            raise ValueError
        ex = vec / L
        cz = e.get('local_z', None)
        if cz is None:
            cand = np.array([0.0, 0.0, 1.0])
            if abs(np.dot(ex, cand)) > 1.0 - 1e-08:
                cand = np.array([0.0, 1.0, 0.0])
            z_raw = cand - np.dot(cand, ex) * ex
            nz = np.linalg.norm(z_raw)
            if nz <= 0.0:
                raise ValueError
            ez = z_raw / nz
        else:
            cz = np.asarray(cz, dtype=float)
            if cz.size != 3:
                raise ValueError
            z_raw = cz - np.dot(cz, ex) * ex
            nz = np.linalg.norm(z_raw)
            if nz < 1e-08:
                cand = np.array([0.0, 0.0, 1.0])
                if abs(np.dot(ex, cand)) > 1.0 - 1e-08:
                    cand = np.array([0.0, 1.0, 0.0])
                z_raw = cand - np.dot(cand, ex) * ex
                nz = np.linalg.norm(z_raw)
                if nz <= 0.0:
                    raise ValueError
            ez = z_raw / nz
        ey = np.cross(ez, ex)
        ny = np.linalg.norm(ey)
        if ny <= 0.0:
            raise ValueError
        ey = ey / ny
        R = np.vstack((ex, ey, ez))
        E = float(e['E'])
        nu = float(e['nu'])
        A = float(e['A'])
        Iy = float(e['I_y'])
        Iz = float(e['I_z'])
        J = float(e['J'])
        if L <= 0.0 or E <= 0.0 or A <= 0.0:
            raise ValueError
        G = E / (2.0 * (1.0 + nu))
        EA = E * A
        GJ = G * J
        EIy = E * Iy
        EIz = E * Iz
        Kloc = np.zeros((12, 12), dtype=float)
        k = EA / L
        Kloc[0, 0] += k
        Kloc[0, 6] -= k
        Kloc[6, 0] -= k
        Kloc[6, 6] += k
        kt = GJ / L
        Kloc[3, 3] += kt
        Kloc[3, 9] -= kt
        Kloc[9, 3] -= kt
        Kloc[9, 9] += kt
        c1 = 12.0 * EIz / L ** 3
        c2 = 6.0 * EIz / L ** 2
        c3 = 4.0 * EIz / L
        c4 = 2.0 * EIz / L
        Kloc[1, 1] += c1
        Kloc[1, 5] += c2
        Kloc[1, 7] += -c1
        Kloc[1, 11] += c2
        Kloc[5, 1] += c2
        Kloc[5, 5] += c3
        Kloc[5, 7] += -c2
        Kloc[5, 11] += c4
        Kloc[7, 1] += -c1
        Kloc[7, 5] += -c2
        Kloc[7, 7] += c1
        Kloc[7, 11] += -c2
        Kloc[11, 1] += c2
        Kloc[11, 5] += c4
        Kloc[11, 7] += -c2
        Kloc[11, 11] += c3
        d1 = 12.0 * EIy / L ** 3
        d2 = 6.0 * EIy / L ** 2
        d3 = 4.0 * EIy / L
        d4 = 2.0 * EIy / L
        Kloc[2, 2] += d1
        Kloc[2, 4] += -d2
        Kloc[2, 8] += -d1
        Kloc[2, 10] += -d2
        Kloc[4, 2] += -d2
        Kloc[4, 4] += d3
        Kloc[4, 8] += d2
        Kloc[4, 10] += d4
        Kloc[8, 2] += -d1
        Kloc[8, 4] += d2
        Kloc[8, 8] += d1
        Kloc[8, 10] += d2
        Kloc[10, 2] += -d2
        Kloc[10, 4] += d4
        Kloc[10, 8] += d2
        Kloc[10, 10] += d3
        T = np.zeros((12, 12), dtype=float)
        T[0:3, 0:3] = R
        T[3:6, 3:6] = R
        T[6:9, 6:9] = R
        T[9:12, 9:12] = R
        Kg = T.T @ Kloc @ T
        dofs_i = node_dofs(i)
        dofs_j = node_dofs(j)
        edofs = np.concatenate((dofs_i, dofs_j))
        K[np.ix_(edofs, edofs)] += Kg
    fixed = np.zeros(ndof, dtype=bool)
    if boundary_conditions is not None:
        for (n, bc) in boundary_conditions.items():
            dofs = node_dofs(int(n))
            bcv = np.asarray(bc, dtype=float).astype(int)
            if bcv.size != 6:
                raise ValueError
            fixed[dofs] = bcv.astype(bool)
    free = ~fixed
    u = np.zeros(ndof, dtype=float)
    r = np.zeros(ndof, dtype=float)
    if np.count_nonzero(free) == 0:
        r[fixed] = -P[fixed]
        return (u, r)
    K_ff = K[np.ix_(free, free)]
    P_f = P[free]
    cond = np.linalg.cond(K_ff)
    if not np.isfinite(cond) or cond >= 1e+16:
        raise ValueError
    u_f = np.linalg.solve(K_ff, P_f)
    u[free] = u_f
    r_full = K @ u - P
    r[fixed] = r_full[fixed]
    return (u, r)