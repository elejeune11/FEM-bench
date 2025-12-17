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
    tiny = 1e-14
    parallel_tol = 1e-08
    cond_tol = 1000000000000.0
    imag_tol_rel = 1e-07
    node_coords = np.asarray(node_coords, dtype=float)
    if node_coords.ndim != 2 or node_coords.shape[1] != 3:
        raise ValueError('node_coords must be an (N,3) array of node coordinates.')
    n_nodes = node_coords.shape[0]
    ndof = 6 * n_nodes
    K = np.zeros((ndof, ndof), dtype=float)
    P = np.zeros(ndof, dtype=float)
    if nodal_loads is not None:
        for n_idx, load in nodal_loads.items():
            if not 0 <= n_idx < n_nodes:
                raise ValueError('nodal_loads contains invalid node index.')
            arr = np.asarray(load, dtype=float)
            if arr.size != 6:
                raise ValueError('Each nodal load entry must have 6 components.')
            P[n_idx * 6:(n_idx + 1) * 6] += arr.reshape(6)
    fixed = np.zeros(ndof, dtype=bool)
    if boundary_conditions is not None:
        for n_idx, bc in boundary_conditions.items():
            if not 0 <= n_idx < n_nodes:
                raise ValueError('boundary_conditions contains invalid node index.')
            bc_arr = np.asarray(bc, dtype=int)
            if bc_arr.size != 6:
                raise ValueError('Each boundary condition must have 6 entries of 0 or 1.')
            for k in range(6):
                if bc_arr[k]:
                    fixed[n_idx * 6 + k] = True
    free = ~fixed
    n_free = int(free.sum())
    if n_free <= 0:
        raise ValueError('All DOFs are fixed; no free DOFs to analyze.')
    ez = np.array([0.0, 0.0, 1.0], dtype=float)
    ey = np.array([0.0, 1.0, 0.0], dtype=float)
    ex = np.array([1.0, 0.0, 0.0], dtype=float)
    elem_cache = []
    for e in elements:
        try:
            i = int(e['node_i'])
            j = int(e['node_j'])
            E = float(e['E'])
            nu = float(e['nu'])
            A = float(e['A'])
            I_y = float(e['I_y'])
            I_z = float(e['I_z'])
            J = float(e['J'])
        except Exception as err:
            raise ValueError('Element properties missing or invalid.') from err
        if not (0 <= i < n_nodes and 0 <= j < n_nodes):
            raise ValueError('Element node indices out of range.')
        xi = node_coords[i]
        xj = node_coords[j]
        dx = xj - xi
        L = float(np.linalg.norm(dx))
        if not np.isfinite(L) or L <= tiny:
            raise ValueError('Element length is zero or invalid.')
        lx = dx / L
        z_hint = None
        if 'local_z' in e and e['local_z'] is not None:
            z_hint = np.asarray(e['local_z'], dtype=float)
        if z_hint is None:
            if abs(np.dot(lx, ez)) >= 1.0 - parallel_tol:
                z_hint = ey.copy()
            else:
                z_hint = ez.copy()
        if np.linalg.norm(z_hint) < tiny or abs(np.dot(lx, z_hint) / (np.linalg.norm(z_hint) + tiny)) >= 1.0 - parallel_tol:
            cand_list = [ey, ez, ex]
            for cand in cand_list:
                if abs(np.dot(lx, cand)) < 1.0 - parallel_tol:
                    z_hint = cand.astype(float).copy()
                    break
            if abs(np.dot(lx, z_hint)) >= 1.0 - parallel_tol:
                if abs(lx[0]) < 0.9:
                    z_hint = np.array([1.0, 0.0, 0.0], dtype=float)
                else:
                    z_hint = np.array([0.0, 1.0, 0.0], dtype=float)
        y_vec = np.cross(z_hint, lx)
        ny = np.linalg.norm(y_vec)
        if ny <= tiny:
            alt = ey if abs(np.dot(lx, ey)) < 1.0 - parallel_tol else ez
            y_vec = np.cross(alt, lx)
            ny = np.linalg.norm(y_vec)
            if ny <= tiny:
                alt = ex
                y_vec = np.cross(alt, lx)
                ny = np.linalg.norm(y_vec)
                if ny <= tiny:
                    raise ValueError('Failed to construct local coordinate system.')
        y_vec = y_vec / ny
        z_vec = np.cross(lx, y_vec)
        nz = np.linalg.norm(z_vec)
        if nz <= tiny:
            raise ValueError('Failed to construct orthonormal local axes.')
        z_vec = z_vec / nz
        R = np.vstack((lx, y_vec, z_vec))
        T = np.zeros((12, 12), dtype=float)
        T[0:3, 0:3] = R
        T[3:6, 3:6] = R
        T[6:9, 6:9] = R
        T[9:12, 9:12] = R
        G = E / (2.0 * (1.0 + nu))
        Ke = np.zeros((12, 12), dtype=float)
        k_ax = E * A / L
        Ke[0, 0] = k_ax
        Ke[0, 6] = -k_ax
        Ke[6, 0] = -k_ax
        Ke[6, 6] = k_ax
        k_tx = G * J / L
        Ke[3, 3] = k_tx
        Ke[3, 9] = -k_tx
        Ke[9, 3] = -k_tx
        Ke[9, 9] = k_tx
        k1 = 12.0 * E * I_z / L ** 3
        k2 = 6.0 * E * I_z / L ** 2
        k3 = 4.0 * E * I_z / L
        k4 = 2.0 * E * I_z / L
        Ke[1, 1] = k1
        Ke[1, 5] = k2
        Ke[1, 7] = -k1
        Ke[1, 11] = k2
        Ke[5, 1] = k2
        Ke[5, 5] = k3
        Ke[5, 7] = -k2
        Ke[5, 11] = k4
        Ke[7, 1] = -k1
        Ke[7, 5] = -k2
        Ke[7, 7] = k1
        Ke[7, 11] = -k2
        Ke[11, 1] = k2
        Ke[11, 5] = k4
        Ke[11, 7] = -k2
        Ke[11, 11] = k3
        kb1 = 12.0 * E * I_y / L ** 3
        kb2 = 6.0 * E * I_y / L ** 2
        kb3 = 4.0 * E * I_y / L
        kb4 = 2.0 * E * I_y / L
        Ke[2, 2] = kb1
        Ke[2, 4] = -kb2
        Ke[2, 8] = -kb1
        Ke[2, 10] = -kb2
        Ke[4, 2] = -kb2
        Ke[4, 4] = kb3
        Ke[4, 8] = kb2
        Ke[4, 10] = kb4
        Ke[8, 2] = -kb1
        Ke[8, 4] = kb2
        Ke[8, 8] = kb1
        Ke[8, 10] = kb2
        Ke[10, 2] = -kb2
        Ke[10, 4] = kb4
        Ke[10, 8] = kb2
        Ke[10, 10] = kb3
        dofs = np.array([i * 6 + 0, i * 6 + 1, i * 6 + 2, i * 6 + 3, i * 6 + 4, i * 6 + 5, j * 6 + 0, j * 6 + 1, j * 6 + 2, j * 6 + 3, j * 6 + 4, j * 6 + 5], dtype=int)
        Kel = T.T @ Ke @ T
        for a in range(12):
            ia = dofs[a]
            Ka_row = Kel[a, :]
            for b in range(12):
                K[ia, dofs[b]] += Ka_row[b]
        elem_cache.append({'dofs': dofs, 'T': T, 'L': L, 'A': A, 'I_y': I_y, 'I_z': I_z, 'Ke_local': Ke})
    Kff = K[np.ix_(free, free)]
    Pff = P[free]
    if Kff.size == 0:
        raise ValueError('No free DOFs after applying boundary conditions.')
    try:
        cond_val = np.linalg.cond(Kff)
    except Exception:
        cond_val = np.inf
    if not np.isfinite(cond_val) or cond_val > cond_tol:
        raise ValueError('Reduced stiffness matrix is singular or ill-conditioned.')
    try:
        u_free = np.linalg.solve(Kff, Pff)
    except np.linalg.LinAlgError as err:
        raise ValueError('Failed to solve linear system; check boundary conditions and connectivity.') from err
    u = np.zeros(ndof, dtype=float)
    u[free] = u_free
    Kg = np.zeros((ndof, ndof), dtype=float)
    for ec in elem_cache:
        dofs = ec['dofs']
        T = ec['T']
        L = ec['L']
        A = ec['A']
        I_y = ec['I_y']
        I_z = ec['I_z']
        Ke_local = ec['Ke_local']
        q_e_global = u[dofs]
        q_local = T @ q_e_global
        s_local = Ke_local @ q_local
        Fx2 = float(s_local[6])
        Mx2 = float(s_local[9])
        My1 = float(s_local[4])
        Mz1 = float(s_local[5])
        My2 = float(s_local[10])
        Mz2 = float(s_local[11])
        I_rho = I_y + I_z
        Kg_local = local_geometric_stiffness_matrix_3D_beam(L=L, A=A, I_rho=I_rho, Fx2=Fx2, Mx2=Mx2, My1=My1, Mz1=Mz1, My2=My2, Mz2=Mz2)
        Kg_global_e = T.T @ Kg_local @ T
        for a in range(12):
            ia = dofs[a]
            Kg_row = Kg_global_e[a, :]
            for b in range(12):
                Kg[ia, dofs[b]] += Kg_row[b]
    Kff = K[np.ix_(free, free)]
    Kgff = Kg[np.ix_(free, free)]
    if np.linalg.norm(Kgff, ord='fro') <= tiny:
        raise ValueError('Geometric stiffness is zero under the provided reference load; cannot perform buckling analysis.')
    try:
        w, vecs = scipy.linalg.eig(Kff, -Kgff)
    except Exception as err:
        raise ValueError('Generalized eigenvalue problem failed to solve.') from err
    w = np.asarray(w)
    abs_real = np.abs(w.real)
    abs_imag = np.abs(w.imag)
    real_mask = abs_imag <= imag_tol_rel * np.maximum(1.0, abs_real)
    w_real = w.real[real_mask]
    vecs_real = vecs[:, real_mask]
    if w_real.size == 0:
        raise ValueError('No real eigenvalues found in buckling analysis.')
    pos_mask = w_real > 0.0
    if not np.any(pos_mask):
        raise ValueError('No positive eigenvalue found; structure may be purely tension-stiffened or improperly constrained.')
    w_pos = w_real[pos_mask]
    vecs_pos = vecs_real[:, pos_mask]
    idx_min = int(np.argmin(w_pos))
    lambda_cr = float(w_pos[idx_min])
    mode_free = vecs_pos[:, idx_min].real
    phi = np.zeros(ndof, dtype=float)
    phi[free] = mode_free
    return (lambda_cr, phi)