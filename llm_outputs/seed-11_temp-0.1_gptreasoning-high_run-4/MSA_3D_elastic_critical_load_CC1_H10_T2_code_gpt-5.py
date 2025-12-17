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

    def _rotation_matrix_and_length(p_i: np.ndarray, p_j: np.ndarray, local_z_opt: Optional[Sequence[float]]):
        dx = p_j - p_i
        L = float(np.linalg.norm(dx))
        if not np.isfinite(L) or L <= 0.0:
            raise ValueError('Element has zero or invalid length.')
        ex = dx / L
        if local_z_opt is None:
            if abs(ex[2]) > 1.0 - 1e-08:
                z_ref = np.array([0.0, 1.0, 0.0], dtype=float)
            else:
                z_ref = np.array([0.0, 0.0, 1.0], dtype=float)
        else:
            z_ref = np.asarray(local_z_opt, dtype=float)
            if z_ref.size != 3:
                raise ValueError('local_z must be length-3 array-like.')
        nz = float(np.linalg.norm(z_ref))
        if not np.isfinite(nz) or nz <= 0.0:
            raise ValueError('local_z must be a finite non-zero vector.')
        z_ref = z_ref / nz
        if abs(float(np.dot(z_ref, ex))) > 1.0 - 1e-08:
            if local_z_opt is not None:
                raise ValueError('Provided local_z is parallel to element axis.')
            candidates = [np.array([0.0, 1.0, 0.0]), np.array([1.0, 0.0, 0.0]), np.array([0.0, 0.0, 1.0])]
            for cand in candidates:
                if abs(float(np.dot(cand, ex))) <= 1.0 - 1e-08:
                    z_ref = cand
                    break
        ey = np.cross(z_ref, ex)
        ney = float(np.linalg.norm(ey))
        if not np.isfinite(ney) or ney <= 1e-12:
            raise ValueError('Failed to construct a valid local frame; check local_z and element alignment.')
        ey = ey / ney
        ez = np.cross(ex, ey)
        R = np.vstack((ex, ey, ez))
        return (R, L)

    def _local_elastic_stiffness(E: float, G: float, A: float, Iy: float, Iz: float, J: float, L: float) -> np.ndarray:
        k = np.zeros((12, 12), dtype=float)
        k_ax = E * A / L
        k[0, 0] += k_ax
        k[0, 6] += -k_ax
        k[6, 0] += -k_ax
        k[6, 6] += k_ax
        k_t = G * J / L
        k[3, 3] += k_t
        k[3, 9] += -k_t
        k[9, 3] += -k_t
        k[9, 9] += k_t
        EIz = E * Iz
        L2 = L * L
        L3 = L2 * L
        i_v1, i_tz1, i_v2, i_tz2 = (1, 5, 7, 11)
        k[i_v1, i_v1] += 12.0 * EIz / L3
        k[i_v1, i_tz1] += 6.0 * EIz / L2
        k[i_v1, i_v2] += -12.0 * EIz / L3
        k[i_v1, i_tz2] += 6.0 * EIz / L2
        k[i_tz1, i_v1] += 6.0 * EIz / L2
        k[i_tz1, i_tz1] += 4.0 * EIz / L
        k[i_tz1, i_v2] += -6.0 * EIz / L2
        k[i_tz1, i_tz2] += 2.0 * EIz / L
        k[i_v2, i_v1] += -12.0 * EIz / L3
        k[i_v2, i_tz1] += -6.0 * EIz / L2
        k[i_v2, i_v2] += 12.0 * EIz / L3
        k[i_v2, i_tz2] += -6.0 * EIz / L2
        k[i_tz2, i_v1] += 6.0 * EIz / L2
        k[i_tz2, i_tz1] += 2.0 * EIz / L
        k[i_tz2, i_v2] += -6.0 * EIz / L2
        k[i_tz2, i_tz2] += 4.0 * EIz / L
        EIy = E * Iy
        i_w1, i_ty1, i_w2, i_ty2 = (2, 4, 8, 10)
        k[i_w1, i_w1] += 12.0 * EIy / L3
        k[i_w1, i_ty1] += -6.0 * EIy / L2
        k[i_w1, i_w2] += -12.0 * EIy / L3
        k[i_w1, i_ty2] += -6.0 * EIy / L2
        k[i_ty1, i_w1] += -6.0 * EIy / L2
        k[i_ty1, i_ty1] += 4.0 * EIy / L
        k[i_ty1, i_w2] += 6.0 * EIy / L2
        k[i_ty1, i_ty2] += 2.0 * EIy / L
        k[i_w2, i_w1] += -12.0 * EIy / L3
        k[i_w2, i_ty1] += 6.0 * EIy / L2
        k[i_w2, i_w2] += 12.0 * EIy / L3
        k[i_w2, i_ty2] += 6.0 * EIy / L2
        k[i_ty2, i_w1] += -6.0 * EIy / L2
        k[i_ty2, i_ty1] += 2.0 * EIy / L
        k[i_ty2, i_w2] += 6.0 * EIy / L2
        k[i_ty2, i_ty2] += 4.0 * EIy / L
        return k
    node_coords = np.asarray(node_coords, dtype=float)
    if node_coords.ndim != 2 or node_coords.shape[1] != 3:
        raise ValueError('node_coords must be an (N,3) array of node coordinates.')
    n_nodes = node_coords.shape[0]
    if n_nodes < 2:
        raise ValueError('At least two nodes are required.')
    ndof = 6 * n_nodes
    K_global = np.zeros((ndof, ndof), dtype=float)
    P_global = np.zeros(ndof, dtype=float)
    if nodal_loads is not None:
        for n, load in nodal_loads.items():
            if n < 0 or n >= n_nodes:
                raise ValueError('nodal_loads refers to an invalid node index.')
            arr = np.asarray(load, dtype=float)
            if arr.size != 6:
                raise ValueError('Each nodal_loads entry must be length-6 [Fx,Fy,Fz,Mx,My,Mz].')
            base = 6 * n
            P_global[base:base + 6] += arr
    for el in elements:
        i = int(el['node_i'])
        j = int(el['node_j'])
        if i < 0 or i >= n_nodes or j < 0 or (j >= n_nodes):
            raise ValueError('Element node indices out of bounds.')
        E = float(el['E'])
        nu = float(el['nu'])
        A = float(el['A'])
        Iy = float(el['I_y'])
        Iz = float(el['I_z'])
        J = float(el['J'])
        local_z_opt = el.get('local_z', None)
        if E <= 0 or A <= 0 or Iy <= 0 or (Iz <= 0) or (J <= 0):
            raise ValueError('Element properties must be positive (E, A, I_y, I_z, J).')
        if not -0.499 < nu < 0.499999:
            raise ValueError("Poisson's ratio must be in (-0.5, 0.5).")
        G = E / (2.0 * (1.0 + nu))
        R, L = _rotation_matrix_and_length(node_coords[i], node_coords[j], local_z_opt)
        k_loc = _local_elastic_stiffness(E, G, A, Iy, Iz, J, L)
        T = np.zeros((12, 12), dtype=float)
        T[0:3, 0:3] = R
        T[3:6, 3:6] = R
        T[6:9, 6:9] = R
        T[9:12, 9:12] = R
        k_glob = T.T @ k_loc @ T
        dof_map = np.array([6 * i + 0, 6 * i + 1, 6 * i + 2, 6 * i + 3, 6 * i + 4, 6 * i + 5, 6 * j + 0, 6 * j + 1, 6 * j + 2, 6 * j + 3, 6 * j + 4, 6 * j + 5], dtype=int)
        for a in range(12):
            ia = dof_map[a]
            K_global[ia, dof_map] += k_glob[a, :]
    fixed = np.zeros(ndof, dtype=bool)
    if boundary_conditions is not None:
        for n, bc in boundary_conditions.items():
            if n < 0 or n >= n_nodes:
                raise ValueError('boundary_conditions refers to an invalid node index.')
            bc_arr = np.asarray(bc, dtype=int).ravel()
            if bc_arr.size != 6 or not np.all((bc_arr == 0) | (bc_arr == 1)):
                raise ValueError('Each boundary_conditions entry must be 6 elements of 0/1.')
            base = 6 * n
            for k in range(6):
                if bc_arr[k] == 1:
                    fixed[base + k] = True
    free = ~fixed
    n_free = int(np.count_nonzero(free))
    if n_free <= 0:
        raise ValueError('All DOFs are constrained; no degrees of freedom to solve.')
    K_ff = K_global[np.ix_(free, free)]
    P_f = P_global[free]
    try:
        u_f = np.linalg.solve(K_ff, P_f)
    except np.linalg.LinAlgError as e:
        raise ValueError('Static solve failed: K_ff is singular or ill-conditioned. Check boundary conditions.') from e
    u_global = np.zeros(ndof, dtype=float)
    u_global[free] = u_f
    K_geo = np.zeros((ndof, ndof), dtype=float)
    for el in elements:
        i = int(el['node_i'])
        j = int(el['node_j'])
        E = float(el['E'])
        nu = float(el['nu'])
        A = float(el['A'])
        Iy = float(el['I_y'])
        Iz = float(el['I_z'])
        J = float(el['J'])
        local_z_opt = el.get('local_z', None)
        G = E / (2.0 * (1.0 + nu))
        R, L = _rotation_matrix_and_length(node_coords[i], node_coords[j], local_z_opt)
        T = np.zeros((12, 12), dtype=float)
        T[0:3, 0:3] = R
        T[3:6, 3:6] = R
        T[6:9, 6:9] = R
        T[9:12, 9:12] = R
        dof_map = np.array([6 * i + 0, 6 * i + 1, 6 * i + 2, 6 * i + 3, 6 * i + 4, 6 * i + 5, 6 * j + 0, 6 * j + 1, 6 * j + 2, 6 * j + 3, 6 * j + 4, 6 * j + 5], dtype=int)
        u_e_glob = u_global[dof_map]
        u_e_loc = T @ u_e_glob
        k_loc = _local_elastic_stiffness(E, G, A, Iy, Iz, J, L)
        f_e_loc = k_loc @ u_e_loc
        Fx2 = float(f_e_loc[6])
        Mx2 = float(f_e_loc[9])
        My1 = float(f_e_loc[4])
        Mz1 = float(f_e_loc[5])
        My2 = float(f_e_loc[10])
        Mz2 = float(f_e_loc[11])
        k_g_loc = local_geometric_stiffness_matrix_3D_beam(L=L, A=A, I_rho=J, Fx2=Fx2, Mx2=Mx2, My1=My1, Mz1=Mz1, My2=My2, Mz2=Mz2)
        k_g_glob = T.T @ k_g_loc @ T
        for a in range(12):
            ia = dof_map[a]
            K_geo[ia, dof_map] += k_g_glob[a, :]
    K_g_ff = K_geo[np.ix_(free, free)]
    try:
        evals, evecs = scipy.linalg.eigh(-K_g_ff, K_ff)
    except Exception as e:
        raise ValueError('Generalized eigenproblem failed.') from e
    tol = 1e-08
    positive_mask = evals > max(tol, 1e-12)
    if not np.any(positive_mask):
        raise ValueError('No positive buckling eigenvalue found (check loading/state).')
    pos_evals = evals[positive_mask]
    pos_evecs = evecs[:, positive_mask]
    min_idx = int(np.argmin(pos_evals))
    lambda_cr = float(pos_evals[min_idx])
    phi_free = pos_evecs[:, min_idx]
    phi_global = np.zeros(ndof, dtype=float)
    phi_global[free] = phi_free
    return (lambda_cr, phi_global)