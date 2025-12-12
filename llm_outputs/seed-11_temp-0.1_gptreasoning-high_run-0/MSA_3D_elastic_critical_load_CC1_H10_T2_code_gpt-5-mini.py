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
    node_coords = np.asarray(node_coords, dtype=float)
    if node_coords.ndim != 2 or node_coords.shape[1] != 3:
        raise ValueError('node_coords must be (N,3) ndarray')
    n_nodes = int(node_coords.shape[0])
    n_dof = n_nodes * 6
    K = np.zeros((n_dof, n_dof), dtype=float)
    K_g = np.zeros((n_dof, n_dof), dtype=float)
    P = np.zeros(n_dof, dtype=float)
    if nodal_loads is not None:
        for (nd, loads) in nodal_loads.items():
            if nd < 0 or nd >= n_nodes:
                raise ValueError('nodal_loads contains invalid node index')
            loads_arr = np.asarray(loads, dtype=float)
            if loads_arr.shape[0] != 6:
                raise ValueError('Each nodal load entry must have 6 components')
            P[6 * nd:6 * nd + 6] += loads_arr

    def bending_base_matrix(L):
        return np.array([[12.0, 6.0 * L, -12.0, 6.0 * L], [6.0 * L, 4.0 * L * L, -6.0 * L, 2.0 * L * L], [-12.0, -6.0 * L, 12.0, -6.0 * L], [6.0 * L, 2.0 * L * L, -6.0 * L, 4.0 * L * L]], dtype=float)
    per_element_info = []
    for el in elements:
        try:
            ni = int(el['node_i'])
            nj = int(el['node_j'])
            E = float(el['E'])
            nu = float(el['nu'])
            A = float(el['A'])
            I_y = float(el['I_y'])
            I_z = float(el['I_z'])
            J = float(el['J'])
        except Exception as exc:
            raise ValueError('Element missing required fields or invalid types') from exc
        if ni < 0 or ni >= n_nodes or nj < 0 or (nj >= n_nodes):
            raise ValueError('Element node index out of range')
        xi = node_coords[ni]
        xj = node_coords[nj]
        x_vec = xj - xi
        L = float(np.linalg.norm(x_vec))
        if L <= 0.0:
            raise ValueError('Element has zero length')
        ex = x_vec / L
        local_z_input = el.get('local_z', None)
        if local_z_input is None:
            global_z = np.array([0.0, 0.0, 1.0], dtype=float)
            global_y = np.array([0.0, 1.0, 0.0], dtype=float)
            if abs(np.dot(ex, global_z)) > 0.999999:
                local_z = global_y.copy()
            else:
                local_z = global_z.copy()
        else:
            local_z = np.asarray(local_z_input, dtype=float)
            if local_z.shape != (3,):
                raise ValueError('local_z must be length-3 vector')
            nz_norm = np.linalg.norm(local_z)
            if nz_norm == 0.0:
                raise ValueError('local_z must be non-zero')
            local_z = local_z / nz_norm
            if np.linalg.norm(np.cross(local_z, ex)) < 1e-08:
                global_z = np.array([0.0, 0.0, 1.0], dtype=float)
                global_y = np.array([0.0, 1.0, 0.0], dtype=float)
                if abs(np.dot(ex, global_z)) > 0.999999:
                    local_z = global_y.copy()
                else:
                    local_z = global_z.copy()
        ey = np.cross(local_z, ex)
        ey_norm = np.linalg.norm(ey)
        if ey_norm < 1e-12:
            alternative = np.array([0.0, 1.0, 0.0], dtype=float)
            ey = np.cross(alternative, ex)
            ey_norm = np.linalg.norm(ey)
            if ey_norm < 1e-12:
                alternative = np.array([1.0, 0.0, 0.0], dtype=float)
                ey = np.cross(alternative, ex)
                ey_norm = np.linalg.norm(ey)
                if ey_norm < 1e-12:
                    raise ValueError('Cannot construct local coordinate system for element')
        ey /= ey_norm
        ez = np.cross(ex, ey)
        ez /= np.linalg.norm(ez)
        R = np.vstack((ex, ey, ez)).astype(float)
        T_e = np.zeros((12, 12), dtype=float)
        T_e[0:3, 0:3] = R
        T_e[3:6, 3:6] = R
        T_e[6:9, 6:9] = R
        T_e[9:12, 9:12] = R
        k_local = np.zeros((12, 12), dtype=float)
        k_ax = E * A / L
        k_local[0, 0] = k_ax
        k_local[0, 6] = -k_ax
        k_local[6, 0] = -k_ax
        k_local[6, 6] = k_ax
        G = E / (2.0 * (1.0 + nu))
        k_tor = G * J / L
        k_local[3, 3] = k_tor
        k_local[3, 9] = -k_tor
        k_local[9, 3] = -k_tor
        k_local[9, 9] = k_tor
        idxs_z = [1, 5, 7, 11]
        ke_b = bending_base_matrix(L)
        k_local[np.ix_(idxs_z, idxs_z)] = E * I_z / L ** 3 * ke_b
        idxs_y = [2, 4, 8, 10]
        k_local[np.ix_(idxs_y, idxs_y)] = E * I_y / L ** 3 * ke_b
        k_global = T_e.T @ k_local @ T_e
        dof_indices = [6 * ni + d for d in range(6)] + [6 * nj + d for d in range(6)]
        dof_indices = np.array(dof_indices, dtype=int)
        K[np.ix_(dof_indices, dof_indices)] += k_global
        per_element_info.append({'dofs': dof_indices, 'k_local': k_local, 'T_e': T_e, 'L': L, 'A': A, 'J': J})
    bc_mask = np.zeros(n_dof, dtype=int)
    if boundary_conditions is not None:
        for (nd, bc_vals) in boundary_conditions.items():
            if nd < 0 or nd >= n_nodes:
                raise ValueError('boundary_conditions contains invalid node index')
            arr = np.asarray(bc_vals, dtype=int)
            if arr.shape[0] != 6:
                raise ValueError('Each boundary condition entry must have 6 components')
            bc_mask[6 * nd:6 * nd + 6] = arr
    free_bool = bc_mask == 0
    free_idx = np.nonzero(free_bool)[0]
    fixed_idx = np.nonzero(~free_bool)[0]
    if free_idx.size == 0:
        raise ValueError('No free degrees of freedom to solve for displacements')
    K_ff = K[np.ix_(free_idx, free_idx)]
    P_f = P[free_idx]
    try:
        u_f = scipy.linalg.solve(K_ff, P_f, assume_a='sym')
    except Exception as exc:
        try:
            u_f = np.linalg.solve(K_ff, P_f)
        except Exception as exc2:
            raise ValueError('Linear system for reference displacements is singular or ill-conditioned') from exc2
    u = np.zeros(n_dof, dtype=float)
    u[free_idx] = np.asarray(u_f, dtype=float)
    for info in per_element_info:
        dof_indices = info['dofs']
        k_local = info['k_local']
        T_e = info['T_e']
        L = info['L']
        A = info['A']
        J = info['J']
        u_e_global = u[dof_indices]
        u_e_local = T_e @ u_e_global
        f_local = k_local @ u_e_local
        Fx2 = float(f_local[6])
        Mx2 = float(f_local[9])
        My1 = float(f_local[4])
        Mz1 = float(f_local[5])
        My2 = float(f_local[10])
        Mz2 = float(f_local[11])
        k_g_local = local_geometric_stiffness_matrix_3D_beam(L, A, J, Fx2, Mx2, My1, Mz1, My2, Mz2)
        k_g_global = T_e.T @ k_g_local @ T_e
        K_g[np.ix_(dof_indices, dof_indices)] += k_g_global
    K_g_ff = K_g[np.ix_(free_idx, free_idx)]
    A_mat = K_ff
    B_mat = -K_g_ff
    try:
        (eigvals, eigvecs) = scipy.linalg.eig(A_mat, B_mat)
    except Exception as exc:
        try:
            invA = np.linalg.inv(A_mat)
            M = invA @ B_mat
            (eigvals2, eigvecs2) = np.linalg.eig(M)
            eigvals = eigvals2
            eigvecs = eigvecs2
        except Exception as exc2:
            raise ValueError('Generalized eigenproblem failed') from exc2
    eigvals = np.asarray(eigvals)
    eigvecs = np.asarray(eigvecs)
    if eigvals.size == 0:
        raise ValueError('Eigenvalue solver returned no eigenvalues')
    imag_tol = 1e-08
    imag_ok = np.abs(np.imag(eigvals)) <= imag_tol * np.maximum(1.0, np.abs(np.real(eigvals)))
    if not np.any(imag_ok):
        raise ValueError('Eigenvalues are significantly complex')
    real_eigvals = np.real(eigvals[imag_ok])
    real_eigvecs = np.real(eigvecs[:, imag_ok])
    pos_tol = 1e-12
    positive_mask = real_eigvals > pos_tol
    if not np.any(positive_mask):
        raise ValueError('No positive eigenvalue found')
    positive_vals = real_eigvals[positive_mask]
    pos_indices = np.nonzero(imag_ok)[0][positive_mask]
    smallest_idx_in_positive = np.argmin(positive_vals)
    chosen_global_idx = pos_indices[smallest_idx_in_positive]
    phi_f = np.real(eigvecs[:, chosen_global_idx])
    phi = np.zeros(n_dof, dtype=float)
    phi[free_idx] = phi_f
    elastic_critical_load_factor = float(np.real(eigvals[chosen_global_idx]))
    return (elastic_critical_load_factor, phi)