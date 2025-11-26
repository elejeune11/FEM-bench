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
    from scipy import linalg
    n_nodes = node_coords.shape[0]
    n_dof = 6 * n_nodes
    K_global = np.zeros((n_dof, n_dof))
    P_global = np.zeros(n_dof)
    for (node_idx, loads) in nodal_loads.items():
        if 0 <= node_idx < n_nodes:
            idx = node_idx * 6
            P_global[idx:idx + 6] = loads
    element_data = []
    for el in elements:
        (i, j) = (el['node_i'], el['node_j'])
        (xi, xj) = (node_coords[i], node_coords[j])
        L_vec = xj - xi
        L = np.linalg.norm(L_vec)
        E = el['E']
        nu = el['nu']
        A = el['A']
        Iy = el['I_y']
        Iz = el['I_z']
        J = el['J']
        G = E / (2.0 * (1.0 + nu))
        k_e = np.zeros((12, 12))
        k_axial = E * A / L
        k_e[0, 0] = k_axial
        k_e[0, 6] = -k_axial
        k_e[6, 0] = -k_axial
        k_e[6, 6] = k_axial
        k_torsion = G * J / L
        k_e[3, 3] = k_torsion
        k_e[3, 9] = -k_torsion
        k_e[9, 3] = -k_torsion
        k_e[9, 9] = k_torsion
        a_z = 12.0 * E * Iz / L ** 3
        b_z = 6.0 * E * Iz / L ** 2
        c_z = 4.0 * E * Iz / L
        d_z = 2.0 * E * Iz / L
        k_e[1, 1] = a_z
        k_e[1, 5] = b_z
        k_e[5, 1] = b_z
        k_e[1, 7] = -a_z
        k_e[7, 1] = -a_z
        k_e[1, 11] = b_z
        k_e[11, 1] = b_z
        k_e[5, 5] = c_z
        k_e[5, 7] = -b_z
        k_e[7, 5] = -b_z
        k_e[5, 11] = d_z
        k_e[11, 5] = d_z
        k_e[7, 7] = a_z
        k_e[7, 11] = -b_z
        k_e[11, 7] = -b_z
        k_e[11, 11] = c_z
        a_y = 12.0 * E * Iy / L ** 3
        b_y = 6.0 * E * Iy / L ** 2
        c_y = 4.0 * E * Iy / L
        d_y = 2.0 * E * Iy / L
        k_e[2, 2] = a_y
        k_e[2, 4] = -b_y
        k_e[4, 2] = -b_y
        k_e[2, 8] = -a_y
        k_e[8, 2] = -a_y
        k_e[2, 10] = -b_y
        k_e[10, 2] = -b_y
        k_e[4, 4] = c_y
        k_e[4, 8] = b_y
        k_e[8, 4] = b_y
        k_e[4, 10] = d_y
        k_e[10, 4] = d_y
        k_e[8, 8] = a_y
        k_e[8, 10] = b_y
        k_e[10, 8] = b_y
        k_e[10, 10] = c_y
        u_vec = L_vec / L
        if 'local_z' in el and el['local_z'] is not None:
            w_ref = np.array(el['local_z'], dtype=float)
        elif abs(u_vec[2]) < 0.999:
            w_ref = np.array([0.0, 0.0, 1.0])
        else:
            w_ref = np.array([0.0, 1.0, 0.0])
        v_vec = np.cross(w_ref, u_vec)
        v_norm = np.linalg.norm(v_vec)
        if v_norm < 1e-10:
            raise ValueError(f'Element {i}-{j}: local_z is parallel to beam axis.')
        v_vec = v_vec / v_norm
        w_vec = np.cross(u_vec, v_vec)
        w_vec = w_vec / np.linalg.norm(w_vec)
        R = np.vstack([u_vec, v_vec, w_vec])
        T = np.zeros((12, 12))
        T[0:3, 0:3] = R
        T[3:6, 3:6] = R
        T[6:9, 6:9] = R
        T[9:12, 9:12] = R
        k_glob_elem = T.T @ k_e @ T
        idx_map = np.r_[i * 6:i * 6 + 6, j * 6:j * 6 + 6]
        for r in range(12):
            row = idx_map[r]
            for c in range(12):
                col = idx_map[c]
                K_global[row, col] += k_glob_elem[r, c]
        element_data.append({'k_e': k_e, 'T': T, 'idx_map': idx_map, 'L': L, 'A': A, 'I_rho': Iy + Iz})
    fixed_mask = np.zeros(n_dof, dtype=bool)
    for (node, constraints) in boundary_conditions.items():
        if 0 <= node < n_nodes:
            base = node * 6
            for (k, val) in enumerate(constraints):
                if val == 1:
                    fixed_mask[base + k] = True
    free_dofs = np.where(~fixed_mask)[0]
    if len(free_dofs) == 0:
        raise ValueError('No free DOFs in the system.')
    K_ff = K_global[np.ix_(free_dofs, free_dofs)]
    P_f = P_global[free_dofs]
    try:
        u_f = linalg.solve(K_ff, P_f)
    except linalg.LinAlgError:
        raise ValueError('System stiffness matrix is singular.')
    u_global = np.zeros(n_dof)
    u_global[free_dofs] = u_f
    Kg_global = np.zeros((n_dof, n_dof))
    for data in element_data:
        k_e = data['k_e']
        T = data['T']
        idx_map = data['idx_map']
        u_ele_glob = u_global[idx_map]
        u_ele_loc = T @ u_ele_glob
        f_loc = k_e @ u_ele_loc
        Fx2 = f_loc[6]
        Mx2 = f_loc[9]
        My1 = f_loc[4]
        Mz1 = f_loc[5]
        My2 = f_loc[10]
        Mz2 = f_loc[11]
        L = data['L']
        A = data['A']
        I_rho = data['I_rho']
        k_g_loc = local_geometric_stiffness_matrix_3D_beam(L, A, I_rho, Fx2, Mx2, My1, Mz1, My2, Mz2)
        k_g_glob_elem = T.T @ k_g_loc @ T
        for r in range(12):
            row = idx_map[r]
            for c in range(12):
                col = idx_map[c]
                Kg_global[row, col] += k_g_glob_elem[r, c]
    Kg_ff = Kg_global[np.ix_(free_dofs, free_dofs)]
    (vals, vecs) = linalg.eig(K_ff, -Kg_ff)
    candidates = []
    for i in range(len(vals)):
        v = vals[i]
        if abs(v.imag) < 1e-06 and v.real > 1e-06:
            candidates.append((v.real, i))
    if not candidates:
        raise ValueError('No positive elastic critical load found.')
    candidates.sort(key=lambda x: x[0])
    (crit_lambda, idx) = candidates[0]
    crit_mode = vecs[:, idx].real
    deformed_shape = np.zeros(n_dof)
    deformed_shape[free_dofs] = crit_mode
    return (crit_lambda, deformed_shape)