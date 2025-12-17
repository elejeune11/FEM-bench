def MSA_3D_elastic_critical_load_CC1_H10_T2(node_coords: np.ndarray, elements: Sequence[dict], boundary_conditions: dict[int, Sequence[int]], nodal_loads: dict[int, Sequence[float]]):
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
    Helper Functions (used here)
    ----------------------------
        Returns the 12x12 local geometric stiffness matrix with torsion-bending coupling for a 3D Euler-Bernoulli beam element.
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

    def _normalize(v: np.ndarray) -> np.ndarray:
        n = np.linalg.norm(v)
        if n == 0.0 or not np.isfinite(n):
            raise ValueError('Zero or invalid vector norm encountered.')
        return v / n

    def _element_rotation_and_length(ni: int, nj: int, local_z_spec) -> tuple[float, np.ndarray]:
        pi = node_coords[ni]
        pj = node_coords[nj]
        dx = pj - pi
        L = float(np.linalg.norm(dx))
        if not np.isfinite(L) or L <= 0.0:
            raise ValueError('Element length must be positive.')
        ex = dx / L
        if local_z_spec is None:
            z_ref = np.array([0.0, 0.0, 1.0], dtype=float)
            if abs(float(np.dot(ex, z_ref))) > 0.99:
                z_ref = np.array([0.0, 1.0, 0.0], dtype=float)
        else:
            z_ref = np.asarray(local_z_spec, dtype=float).reshape(3)
            if not np.all(np.isfinite(z_ref)):
                raise ValueError('Invalid local_z vector.')
        if abs(float(np.dot(ex, z_ref))) > 0.999:
            candidates = [np.array([0.0, 0.0, 1.0]), np.array([0.0, 1.0, 0.0]), np.array([1.0, 0.0, 0.0])]
            for cand in candidates:
                if abs(float(np.dot(ex, cand))) < 0.9:
                    z_ref = cand.astype(float)
                    break
        ey = _normalize(np.cross(z_ref, ex))
        ez = np.cross(ex, ey)
        ez = _normalize(ez)
        ey = np.cross(ez, ex)
        ey = _normalize(ey)
        R = np.column_stack((ex, ey, ez))
        return (L, R)

    def _local_elastic_stiffness(E: float, nu: float, A: float, Iy: float, Iz: float, J: float, L: float) -> np.ndarray:
        k = np.zeros((12, 12), dtype=float)
        EA_L = E * A / L
        G = E / (2.0 * (1.0 + nu))
        GJ_L = G * J / L
        EIy = E * Iy
        EIz = E * Iz
        k[0, 0] = EA_L
        k[0, 6] = -EA_L
        k[6, 0] = -EA_L
        k[6, 6] = EA_L
        k[3, 3] = GJ_L
        k[3, 9] = -GJ_L
        k[9, 3] = -GJ_L
        k[9, 9] = GJ_L
        a = 12.0 * EIz / L ** 3
        b = 6.0 * EIz / L ** 2
        c = 4.0 * EIz / L
        d = 2.0 * EIz / L
        k[1, 1] += a
        k[1, 5] += b
        k[1, 7] += -a
        k[1, 11] += b
        k[5, 1] += b
        k[5, 5] += c
        k[5, 7] += -b
        k[5, 11] += d
        k[7, 1] += -a
        k[7, 5] += -b
        k[7, 7] += a
        k[7, 11] += -b
        k[11, 1] += b
        k[11, 5] += d
        k[11, 7] += -b
        k[11, 11] += c
        a = 12.0 * EIy / L ** 3
        b = 6.0 * EIy / L ** 2
        c = 4.0 * EIy / L
        d = 2.0 * EIy / L
        k[2, 2] += a
        k[2, 4] += -b
        k[2, 8] += -a
        k[2, 10] += -b
        k[4, 2] += -b
        k[4, 4] += c
        k[4, 8] += b
        k[4, 10] += d
        k[8, 2] += -a
        k[8, 4] += b
        k[8, 8] += a
        k[8, 10] += b
        k[10, 2] += -b
        k[10, 4] += d
        k[10, 8] += b
        k[10, 10] += c
        return k
    n_nodes = int(node_coords.shape[0])
    if node_coords.ndim != 2 or node_coords.shape[1] != 3:
        raise ValueError('node_coords must be (N,3) array.')
    ndof = 6 * n_nodes
    K = np.zeros((ndof, ndof), dtype=float)
    Kg = np.zeros((ndof, ndof), dtype=float)
    P = np.zeros(ndof, dtype=float)
    for ni, load in nodal_loads.items():
        idx0 = 6 * int(ni)
        vec = np.asarray(load, dtype=float).reshape(6)
        P[idx0:idx0 + 6] += vec
    fixed = np.zeros(ndof, dtype=bool)
    for ni, bc in boundary_conditions.items():
        mask = np.asarray(bc, dtype=int).reshape(6)
        idx0 = 6 * int(ni)
        fixed[idx0:idx0 + 6] = mask.astype(bool)
    free = ~fixed
    if free.sum() == 0:
        raise ValueError('All DOFs are constrained; no free DOFs.')
    element_data = []
    for e in elements:
        ni = int(e['node_i'])
        nj = int(e['node_j'])
        E = float(e['E'])
        nu = float(e['nu'])
        A = float(e['A'])
        Iy = float(e['I_y'])
        Iz = float(e['I_z'])
        J = float(e['J'])
        local_z = e.get('local_z', None)
        L, R = _element_rotation_and_length(ni, nj, local_z)
        k_loc = _local_elastic_stiffness(E, nu, A, Iy, Iz, J, L)
        T = np.zeros((12, 12), dtype=float)
        for blk in range(4):
            T[3 * blk:3 * (blk + 1), 3 * blk:3 * (blk + 1)] = R
        dofs = np.concatenate([np.arange(6 * ni, 6 * ni + 6), np.arange(6 * nj, 6 * nj + 6)]).astype(int)
        Ke = T @ k_loc @ T.T
        K[np.ix_(dofs, dofs)] += Ke
        element_data.append((ni, nj, L, R, T, k_loc, A, J))
    K_ff = K[np.ix_(free, free)]
    P_ff = P[free]
    try:
        u_ff = scipy.linalg.solve(K_ff, P_ff, assume_a='sym')
    except Exception as exc:
        raise ValueError('Failed to solve for static displacements; check BCs and stiffness.') from exc
    u = np.zeros(ndof, dtype=float)
    u[free] = u_ff
    for ni, nj, L, R, T, k_loc, A, J in element_data:
        dofs = np.concatenate([np.arange(6 * ni, 6 * ni + 6), np.arange(6 * nj, 6 * nj + 6)]).astype(int)
        u_e_glob = u[dofs]
        u_loc = T.T @ u_e_glob
        f_loc = k_loc @ u_loc
        Fx2 = float(f_loc[6])
        Mx2 = float(f_loc[9])
        My1 = float(f_loc[4])
        Mz1 = float(f_loc[5])
        My2 = float(f_loc[10])
        Mz2 = float(f_loc[11])
        k_g_loc = local_geometric_stiffness_matrix_3D_beam(L, A, J, Fx2, Mx2, My1, Mz1, My2, Mz2)
        Kg_e = T @ k_g_loc @ T.T
        Kg[np.ix_(dofs, dofs)] += Kg_e
    Kg_ff = Kg[np.ix_(free, free)]
    norm_K = np.linalg.norm(K_ff, ord='fro')
    norm_Kg = np.linalg.norm(Kg_ff, ord='fro')
    if not np.isfinite(norm_Kg) or norm_Kg < 1e-14 * max(1.0, norm_K):
        raise ValueError('Geometric stiffness is near zero under the reference load; buckling factor undefined.')
    A = -K_ff
    B = Kg_ff
    try:
        eigvals, eigvecs = scipy.linalg.eig(A, B)
    except Exception as exc:
        raise ValueError('Generalized eigenvalue solution failed.') from exc
    eigvals = np.asarray(eigvals)
    eigvecs = np.asarray(eigvecs)
    tol_im = 1e-08
    real_mask = np.isfinite(eigvals.real) & np.isfinite(eigvals.imag) & (np.abs(eigvals.imag) <= tol_im * np.maximum(1.0, np.abs(eigvals.real)))
    pos_mask = eigvals.real > 1e-12
    valid = np.where(real_mask & pos_mask)[0]
    if valid.size == 0:
        raise ValueError('No positive real eigenvalue found for buckling.')
    vals_real = eigvals.real[valid]
    idx_min_local = int(np.argmin(vals_real))
    idx_global = int(valid[idx_min_local])
    lam = float(vals_real[idx_min_local])
    phi_ff = eigvecs[:, idx_global].real
    phi = np.zeros(ndof, dtype=float)
    phi[free] = phi_ff
    return (lam, phi)