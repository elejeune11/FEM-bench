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

    def _validate_and_prepare_inputs(node_coords_in, elements_in, bcs_in, loads_in):
        coords = np.asarray(node_coords_in, dtype=float)
        if coords.ndim != 2 or coords.shape[1] != 3:
            raise ValueError('node_coords must be an (N,3) array')
        n_nodes_local = coords.shape[0]
        ndof_local = 6 * n_nodes_local
        fixed = np.zeros(ndof_local, dtype=bool)
        if bcs_in is not None:
            for (n, flags) in bcs_in.items():
                if n < 0 or n >= n_nodes_local:
                    raise ValueError('Boundary condition node index out of range')
                flags_arr = np.asarray(list(flags), dtype=int)
                if flags_arr.size != 6:
                    raise ValueError('Each boundary condition must have 6 entries')
                if np.any((flags_arr != 0) & (flags_arr != 1)):
                    raise ValueError('Boundary condition entries must be 0 or 1')
                fixed[n * 6:(n + 1) * 6] = flags_arr.astype(bool)
        P = np.zeros(ndof_local, dtype=float)
        if loads_in is not None:
            for (n, l) in loads_in.items():
                if n < 0 or n >= n_nodes_local:
                    raise ValueError('Load node index out of range')
                l_arr = np.asarray(list(l), dtype=float)
                if l_arr.size != 6:
                    raise ValueError('Each load must have 6 entries')
                P[n * 6:(n + 1) * 6] += l_arr
        elems = []
        for el in elements_in:
            if not all((k in el for k in ('node_i', 'node_j', 'E', 'nu', 'A', 'I_y', 'I_z', 'J'))):
                raise ValueError("Each element must include keys 'node_i','node_j','E','nu','A','I_y','I_z','J'")
            ni = int(el['node_i'])
            nj = int(el['node_j'])
            if ni < 0 or ni >= n_nodes_local or nj < 0 or (nj >= n_nodes_local) or (ni == nj):
                raise ValueError('Invalid element node indices')
            E = float(el['E'])
            nu = float(el['nu'])
            A = float(el['A'])
            Iy = float(el['I_y'])
            Iz = float(el['I_z'])
            J = float(el['J'])
            locz = el.get('local_z', None)
            if locz is not None:
                locz = np.asarray(locz, dtype=float)
                if locz.shape != (3,):
                    raise ValueError('local_z must be a length-3 vector')
            elems.append({'node_i': ni, 'node_j': nj, 'E': E, 'nu': nu, 'A': A, 'I_y': Iy, 'I_z': Iz, 'J': J, 'local_z': locz})
        return (coords, elems, fixed, P)

    def _rotation_matrix_and_length(xi, xj, local_z):
        dx = xj - xi
        L = float(np.linalg.norm(dx))
        if not np.isfinite(L) or L <= 0.0:
            raise ValueError('Element length must be positive')
        ex = dx / L
        if local_z is None:
            ref = np.array([0.0, 0.0, 1.0])
            if abs(np.dot(ex, ref)) > 0.99:
                ref = np.array([0.0, 1.0, 0.0])
        else:
            ref = np.array(local_z, dtype=float)
            nref = float(np.linalg.norm(ref))
            if nref == 0.0 or not np.isfinite(nref):
                raise ValueError('local_z must be a non-zero finite vector')
            ref = ref / nref
            if abs(np.dot(ex, ref)) >= 1.0 - 1e-10:
                raise ValueError('Provided local_z is parallel to the element axis')
        ey_temp = np.cross(ex, ref)
        ny = float(np.linalg.norm(ey_temp))
        if ny < 1e-12:
            alt = np.array([0.0, 1.0, 0.0]) if abs(np.dot(ex, np.array([0.0, 1.0, 0.0]))) < 0.99 else np.array([1.0, 0.0, 0.0])
            ey_temp = np.cross(ex, alt)
            ny = float(np.linalg.norm(ey_temp))
            if ny < 1e-12:
                raise ValueError('Cannot determine a valid local coordinate system')
        ey = ey_temp / ny
        ez = np.cross(ey, ex)
        R = np.vstack((ex, ey, ez))
        return (R, L)

    def _T_from_R(R):
        T = np.zeros((12, 12), dtype=float)
        T[0:3, 0:3] = R
        T[3:6, 3:6] = R
        T[6:9, 6:9] = R
        T[9:12, 9:12] = R
        return T

    def _local_elastic_stiffness(E, G, A, Iy, Iz, J, L):
        k = np.zeros((12, 12), dtype=float)
        a = E * A / L
        k[0, 0] = a
        k[0, 6] = -a
        k[6, 0] = -a
        k[6, 6] = a
        t = G * J / L
        k[3, 3] = t
        k[3, 9] = -t
        k[9, 3] = -t
        k[9, 9] = t
        EIz = E * Iz
        c1 = 12.0 * EIz / L ** 3
        c2 = 6.0 * EIz / L ** 2
        c3 = 4.0 * EIz / L
        c4 = 2.0 * EIz / L
        k[1, 1] += c1
        k[1, 5] += c2
        k[1, 7] += -c1
        k[1, 11] += c2
        k[5, 1] += c2
        k[5, 5] += c3
        k[5, 7] += -c2
        k[5, 11] += c4
        k[7, 1] += -c1
        k[7, 5] += -c2
        k[7, 7] += c1
        k[7, 11] += -c2
        k[11, 1] += c2
        k[11, 5] += c4
        k[11, 7] += -c2
        k[11, 11] += c3
        EIy = E * Iy
        d1 = 12.0 * EIy / L ** 3
        d2 = 6.0 * EIy / L ** 2
        d3 = 4.0 * EIy / L
        d4 = 2.0 * EIy / L
        k[2, 2] += d1
        k[2, 4] += -d2
        k[2, 8] += -d1
        k[2, 10] += -d2
        k[4, 2] += -d2
        k[4, 4] += d3
        k[4, 8] += d2
        k[4, 10] += d4
        k[8, 2] += -d1
        k[8, 4] += d2
        k[8, 8] += d1
        k[8, 10] += d2
        k[10, 2] += -d2
        k[10, 4] += d4
        k[10, 8] += d2
        k[10, 10] += d3
        return k

    def _element_dof_indices(ni, nj):
        idx_i = np.arange(ni * 6, ni * 6 + 6, dtype=int)
        idx_j = np.arange(nj * 6, nj * 6 + 6, dtype=int)
        return np.concatenate((idx_i, idx_j))
    (coords, elems, fixed_mask, P) = _validate_and_prepare_inputs(node_coords, elements, boundary_conditions, nodal_loads)
    n_nodes = coords.shape[0]
    ndof = 6 * n_nodes
    K = np.zeros((ndof, ndof), dtype=float)
    elem_cache = []
    for el in elems:
        ni = el['node_i']
        nj = el['node_j']
        xi = coords[ni]
        xj = coords[nj]
        (R, L) = _rotation_matrix_and_length(xi, xj, el['local_z'])
        T = _T_from_R(R)
        E = el['E']
        nu = el['nu']
        A = el['A']
        Iy = el['I_y']
        Iz = el['I_z']
        J = el['J']
        if not np.isfinite(E) or not np.isfinite(nu) or (not np.isfinite(A)) or (not np.isfinite(Iy)) or (not np.isfinite(Iz)) or (not np.isfinite(J)):
            raise ValueError('Element properties must be finite numbers')
        G = E / (2.0 * (1.0 + nu))
        k_local = _local_elastic_stiffness(E, G, A, Iy, Iz, J, L)
        k_global = T.T @ k_local @ T
        edofs = _element_dof_indices(ni, nj)
        K[np.ix_(edofs, edofs)] += k_global
        elem_cache.append({'ni': ni, 'nj': nj, 'R': R, 'T': T, 'L': L, 'E': E, 'G': G, 'A': A, 'Iy': Iy, 'Iz': Iz, 'J': J, 'k_local': k_local})
    free_mask = ~fixed_mask
    n_free = int(np.count_nonzero(free_mask))
    if n_free <= 0:
        raise ValueError('All degrees of freedom are fixed; cannot solve')
    K_ff = K[np.ix_(free_mask, free_mask)]
    P_f = P[free_mask]
    K_ff = 0.5 * (K_ff + K_ff.T)
    try:
        u_f = np.linalg.solve(K_ff, P_f)
    except np.linalg.LinAlgError as e:
        raise ValueError('Reduced stiffness matrix is singular or ill-conditioned') from e
    u = np.zeros(ndof, dtype=float)
    u[free_mask] = u_f
    K_g = np.zeros((ndof, ndof), dtype=float)
    for ec in elem_cache:
        ni = ec['ni']
        nj = ec['nj']
        T = ec['T']
        L = ec['L']
        A = ec['A']
        Iy = ec['Iy']
        Iz = ec['Iz']
        k_local = ec['k_local']
        edofs = _element_dof_indices(ni, nj)
        d_e_global = u[edofs]
        d_local = T @ d_e_global
        f_local = k_local @ d_local
        Fx2 = float(f_local[6])
        Mx2 = float(f_local[9])
        My1 = float(f_local[4])
        Mz1 = float(f_local[5])
        My2 = float(f_local[10])
        Mz2 = float(f_local[11])
        I_rho = Iy + Iz
        k_g_local = local_geometric_stiffness_matrix_3D_beam(L, A, I_rho, Fx2, Mx2, My1, Mz1, My2, Mz2)
        k_g_global = T.T @ k_g_local @ T
        K_g[np.ix_(edofs, edofs)] += k_g_global
    K_g_ff = K_g[np.ix_(free_mask, free_mask)]
    K_g_ff = 0.5 * (K_g_ff + K_g_ff.T)
    A_mat = K_ff
    B_mat = -K_g_ff
    try:
        (evals, evecs) = scipy.linalg.eig(A_mat, B_mat)
    except Exception as e:
        raise ValueError('Generalized eigenvalue problem failed') from e
    evals = np.asarray(evals)
    evecs = np.asarray(evecs)
    imag_tol = 1e-08
    pos_tol = 1e-10
    real_mask = np.abs(evals.imag) <= imag_tol * np.maximum(1.0, np.abs(evals.real))
    evals_real = evals.real[real_mask]
    evecs_real = evecs[:, real_mask]
    pos_mask = evals_real > pos_tol
    if not np.any(pos_mask):
        raise ValueError('No positive real eigenvalues found for buckling problem')
    evals_pos = evals_real[pos_mask]
    evecs_pos = evecs_real[:, pos_mask]
    min_idx = int(np.argmin(evals_pos))
    lambda_cr = float(evals_pos[min_idx])
    phi_f = evecs_pos[:, min_idx].real
    phi = np.zeros(ndof, dtype=float)
    phi[free_mask] = phi_f
    return (lambda_cr, phi)