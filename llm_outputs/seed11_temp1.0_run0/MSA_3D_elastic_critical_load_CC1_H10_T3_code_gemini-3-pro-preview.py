def MSA_3D_elastic_critical_load_CC1_H10_T3(node_coords: np.ndarray, elements: Sequence[dict], boundary_conditions: dict[int, Sequence[int]], nodal_loads: dict[int, Sequence[float]]):
    """
    Perform linear (eigenvalue) buckling analysis for a 3D frame and return the
    elastic critical load factor and associated global buckling mode shape.
    Overview
    --------
    The routine:
      1) Assembles the global elastic stiffness matrix `K`.
      2) Assembles the global reference load vector `P`.
      3) Solves the linear static problem `K u = P` (with boundary conditions) to
         obtain the displacement state `u` under the reference load.
      4) Assembles the geometric stiffness `K_g` consistent with that state.
      5) Solves the generalized eigenproblem on the free DOFs,
             K φ = -λ K_g φ,
         and returns the smallest positive eigenvalue `λ` as the elastic
         critical load factor and its corresponding global mode shape `φ`
         (constrained DOFs set to zero).
    Parameters
    ----------
    node_coords : (N, 3) float ndarray
        Global coordinates of the N nodes (row i → [x, y, z] of node i, 0-based).
    elements : iterable of dict
        Each dict must contain:
            'node_i', 'node_j' : int
                End node indices (0-based).
            'E', 'nu', 'A', 'I_y', 'I_z', 'J' : float
                Material and geometric properties.
            'local_z' : array-like of shape (3,), optional
                Unit vector in global coordinates defining the local z-axis orientation.
                Must not be parallel to the beam axis. If None, a default is chosen.
                Default local_z = global z, unless the beam lies along global z — then default local_z = global y.
    boundary_conditions : dict[int, Sequence[int]]
        Maps node index to a 6-element iterable of 0 (free) or 1 (fixed) values.
        Omitted nodes are assumed to have all DOFs free.
    nodal_loads : dict[int, Sequence[float]]
        Maps node index to a 6-element array of applied loads:
        [Fx, Fy, Fz, Mx, My, Mz]. Omitted nodes are assumed to have zero loads.
    Returns
    -------
    elastic_critical_load_factor : float
        The smallest positive eigenvalue `λ` (> 0). If `P` is the reference load
        used to form `K_g`, then the predicted elastic buckling load is
        `P_cr = λ · P`.
    deformed_shape_vector : (6*n_nodes,) ndarray of float
        Global buckling mode vector with constrained DOFs set to zero. No
        normalization is applied (mode scale is arbitrary; only the shape matters).
    Assumptions
    -----------
      `[u_x, u_y, u_z, θ_x, θ_y, θ_z]`.
      represented via `K_g` assembled at the reference load state, not via a full
      nonlinear equilibrium/path-following analysis.
    Raises
    ------
    ValueError
        Propagated from called routines if:
    Notes
    -----
      if desired (e.g., by max absolute translational DOF).
      the returned mode can depend on numerical details.
        + **Tension (+Fx2)** increases lateral/torsional stiffness.
        + Compression (-Fx2) decreases it and may trigger buckling when K_e + K_g becomes singular.
    """
    import numpy as np
    import scipy.linalg
    n_nodes = node_coords.shape[0]
    n_dof = n_nodes * 6

    def get_elem_geom(p1, p2, local_z):
        dp = p2 - p1
        L = np.linalg.norm(dp)
        if L < 1e-12:
            raise ValueError('Element length is near zero.')
        x_vec = dp / L
        if local_z is None:
            if np.abs(x_vec[2]) > 0.999999:
                lz = np.array([0.0, 1.0, 0.0])
            else:
                lz = np.array([0.0, 0.0, 1.0])
        else:
            lz = np.array(local_z, dtype=float)
        y_vec = np.cross(lz, x_vec)
        norm_y = np.linalg.norm(y_vec)
        if norm_y < 1e-12:
            raise ValueError('Local z vector is parallel to element axis.')
        y_vec /= norm_y
        z_vec = np.cross(x_vec, y_vec)
        z_vec /= np.linalg.norm(z_vec)
        R = np.vstack([x_vec, y_vec, z_vec])
        return (R, L)
    K_global = np.zeros((n_dof, n_dof))
    elem_store = []
    for el in elements:
        (i, j) = (el['node_i'], el['node_j'])
        (p1, p2) = (node_coords[i], node_coords[j])
        (R3, L) = get_elem_geom(p1, p2, el.get('local_z'))
        T = np.zeros((12, 12))
        for block in range(4):
            T[3 * block:3 * block + 3, 3 * block:3 * block + 3] = R3
        E = el['E']
        nu = el['nu']
        A = el['A']
        Iy = el['I_y']
        Iz = el['I_z']
        J = el['J']
        G = E / (2.0 * (1.0 + nu))
        k_e = np.zeros((12, 12))
        ka = E * A / L
        k_e[0, 0] = ka
        k_e[0, 6] = -ka
        k_e[6, 0] = -ka
        k_e[6, 6] = ka
        kt = G * J / L
        k_e[3, 3] = kt
        k_e[3, 9] = -kt
        k_e[9, 3] = -kt
        k_e[9, 9] = kt
        k_bz = E * Iz / L ** 3
        bz_mat = np.array([[12, 6 * L, -12, 6 * L], [6 * L, 4 * L ** 2, -6 * L, 2 * L ** 2], [-12, -6 * L, 12, -6 * L], [6 * L, 2 * L ** 2, -6 * L, 4 * L ** 2]])
        idx_z = [1, 5, 7, 11]
        for r in range(4):
            for c in range(4):
                k_e[idx_z[r], idx_z[c]] = k_bz * bz_mat[r, c]
        k_by = E * Iy / L ** 3
        by_mat = np.array([[12, -6 * L, -12, -6 * L], [-6 * L, 4 * L ** 2, 6 * L, 2 * L ** 2], [-12, 6 * L, 12, 6 * L], [-6 * L, 2 * L ** 2, 6 * L, 4 * L ** 2]])
        idx_y = [2, 4, 8, 10]
        for r in range(4):
            for c in range(4):
                k_e[idx_y[r], idx_y[c]] = k_by * by_mat[r, c]
        k_g_elem = T.T @ k_e @ T
        indices = []
        for node_idx in [i, j]:
            for d in range(6):
                indices.append(node_idx * 6 + d)
        for r in range(12):
            row = indices[r]
            for c in range(12):
                col = indices[c]
                K_global[row, col] += k_g_elem[r, c]
        elem_store.append({'nodes': [i, j], 'indices': indices, 'T': T, 'ke': k_e, 'L': L, 'props': el})
    P_global = np.zeros(n_dof)
    for (n_idx, loads) in nodal_loads.items():
        for d in range(6):
            P_global[n_idx * 6 + d] = loads[d]
    is_fixed = np.zeros(n_dof, dtype=bool)
    for (n_idx, bcs) in boundary_conditions.items():
        for d in range(6):
            if bcs[d] == 1:
                is_fixed[n_idx * 6 + d] = True
    free_dofs = np.where(~is_fixed)[0]
    if len(free_dofs) == 0:
        raise ValueError('No free degrees of freedom.')
    K_ff = K_global[np.ix_(free_dofs, free_dofs)]
    P_f = P_global[free_dofs]
    try:
        u_f = scipy.linalg.solve(K_ff, P_f)
    except scipy.linalg.LinAlgError:
        raise ValueError('Singular stiffness matrix in reference static solve.')
    u_global = np.zeros(n_dof)
    u_global[free_dofs] = u_f
    Kg_global = np.zeros((n_dof, n_dof))
    for item in elem_store:
        T = item['T']
        ke = item['ke']
        L = item['L']
        props = item['props']
        indices = item['indices']
        u_elem = u_global[indices]
        u_loc = T @ u_elem
        f_int = ke @ u_loc
        Fx2 = f_int[6]
        P = Fx2
        kg_loc = np.zeros((12, 12))
        c = P / (30.0 * L)
        m1 = np.array([[36, 3 * L, -36, 3 * L], [3 * L, 4 * L ** 2, -3 * L, -L ** 2], [-36, -3 * L, 36, -3 * L], [3 * L, -L ** 2, -3 * L, 4 * L ** 2]])
        idx_1 = [1, 5, 7, 11]
        for r in range(4):
            for c_i in range(4):
                kg_loc[idx_1[r], idx_1[c_i]] += c * m1[r, c_i]
        m2 = np.array([[36, -3 * L, -36, -3 * L], [-3 * L, 4 * L ** 2, 3 * L, -L ** 2], [-36, 3 * L, 36, 3 * L], [-3 * L, -L ** 2, 3 * L, 4 * L ** 2]])
        idx_2 = [2, 4, 8, 10]
        for r in range(4):
            for c_i in range(4):
                kg_loc[idx_2[r], idx_2[c_i]] += c * m2[r, c_i]
        Ip = props['I_y'] + props['I_z']
        r2 = Ip / props['A']
        kt = P * r2 / L
        kg_loc[3, 3] += kt
        kg_loc[3, 9] -= kt
        kg_loc[9, 3] -= kt
        kg_loc[9, 9] += kt
        kg_glob_elem = T.T @ kg_loc @ T
        for r in range(12):
            g_r = indices[r]
            for c_i in range(12):
                g_c = indices[c_i]
                Kg_global[g_r, g_c] += kg_glob_elem[r, c_i]
    Kg_ff = Kg_global[np.ix_(free_dofs, free_dofs)]
    if np.all(np.abs(Kg_ff) < 1e-15):
        raise ValueError('Geometric stiffness matrix is zero (no internal forces).')
    try:
        (vals, vecs) = scipy.linalg.eig(K_ff, -Kg_ff)
    except Exception:
        raise ValueError('Eigenvalue solver failed.')
    valid_eigs = []
    for i in range(len(vals)):
        v = vals[i]
        if abs(v.imag) < 1e-06 and v.real > 1e-09:
            valid_eigs.append((v.real, i))
    if not valid_eigs:
        raise ValueError('No positive elastic critical load factor found.')
    valid_eigs.sort(key=lambda x: x[0])
    (lambda_cr, idx) = valid_eigs[0]
    shape_f = vecs[:, idx]
    deformed_shape = np.zeros(n_dof)
    deformed_shape[free_dofs] = np.real(shape_f)
    return (float(lambda_cr), deformed_shape)