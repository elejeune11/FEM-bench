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
    n_nodes = node_coords.shape[0]
    n_dofs = n_nodes * 6
    K = np.zeros((n_dofs, n_dofs))
    P = np.zeros(n_dofs)
    for (node_idx, loads) in nodal_loads.items():
        start_dof = node_idx * 6
        P[start_dof:start_dof + 6] = loads
    element_data_cache = []
    for el in elements:
        (i, j) = (el['node_i'], el['node_j'])
        node_i_coords = node_coords[i]
        node_j_coords = node_coords[j]
        (E, nu, A) = (el['E'], el['nu'], el['A'])
        (Iy, Iz, J) = (el['I_y'], el['I_z'], el['J'])
        G = E / (2 * (1 + nu))
        vec_ij = node_j_coords - node_i_coords
        L = np.linalg.norm(vec_ij)
        if np.isclose(L, 0):
            continue
        x_local = vec_ij / L
        if el.get('local_z') is not None:
            z_prime = np.asarray(el['local_z'])
        else:
            global_z = np.array([0.0, 0.0, 1.0])
            if np.allclose(np.abs(np.dot(x_local, global_z)), 1.0):
                z_prime = np.array([0.0, 1.0, 0.0])
            else:
                z_prime = global_z
        y_local = np.cross(z_prime, x_local)
        y_local /= np.linalg.norm(y_local)
        z_local = np.cross(x_local, y_local)
        z_local /= np.linalg.norm(z_local)
        R = np.array([x_local, y_local, z_local])
        T_rot = scipy.linalg.block_diag(R, R, R, R)
        k_e = np.zeros((12, 12))
        k_e[0, 0] = k_e[6, 6] = E * A / L
        k_e[0, 6] = k_e[6, 0] = -E * A / L
        k_e[3, 3] = k_e[9, 9] = G * J / L
        k_e[3, 9] = k_e[9, 3] = -G * J / L
        (EIz_L3, EIz_L2, EIz_L) = (E * Iz / L ** 3, E * Iz / L ** 2, E * Iz / L)
        k_e_yz = np.array([[12 * EIz_L3, 6 * EIz_L2, -12 * EIz_L3, 6 * EIz_L2], [6 * EIz_L2, 4 * EIz_L, -6 * EIz_L2, 2 * EIz_L], [-12 * EIz_L3, -6 * EIz_L2, 12 * EIz_L3, -6 * EIz_L2], [6 * EIz_L2, 2 * EIz_L, -6 * EIz_L2, 4 * EIz_L]])
        indices_yz = np.ix_([1, 5, 7, 11], [1, 5, 7, 11])
        k_e[indices_yz] += k_e_yz
        (EIy_L3, EIy_L2, EIy_L) = (E * Iy / L ** 3, E * Iy / L ** 2, E * Iy / L)
        k_e_xz = np.array([[12 * EIy_L3, -6 * EIy_L2, -12 * EIy_L3, -6 * EIy_L2], [-6 * EIy_L2, 4 * EIy_L, 6 * EIy_L2, 2 * EIy_L], [-12 * EIy_L3, 6 * EIy_L2, 12 * EIy_L3, 6 * EIy_L2], [-6 * EIy_L2, 2 * EIy_L, 6 * EIy_L2, 4 * EIy_L]])
        indices_xz = np.ix_([2, 4, 8, 10], [2, 4, 8, 10])
        k_e[indices_xz] += k_e_xz
        K_e = T_rot.T @ k_e @ T_rot
        dofs = np.concatenate((np.arange(i * 6, i * 6 + 6), np.arange(j * 6, j * 6 + 6)))
        K[np.ix_(dofs, dofs)] += K_e
        element_data_cache.append({'dofs': dofs, 'T_rot': T_rot, 'k_e': k_e, 'L': L, 'A': A, 'Iy': Iy, 'Iz': Iz})
    all_dofs = np.arange(n_dofs)
    constrained_dofs = []
    for (node_idx, bc_flags) in boundary_conditions.items():
        for (dof_idx, flag) in enumerate(bc_flags):
            if flag == 1:
                constrained_dofs.append(node_idx * 6 + dof_idx)
    free_dofs = np.setdiff1d(all_dofs, constrained_dofs, assume_unique=True)
    K_ff = K[np.ix_(free_dofs, free_dofs)]
    P_f = P[free_dofs]
    try:
        u_f = scipy.linalg.solve(K_ff, P_f, assume_a='sym')
    except (np.linalg.LinAlgError, ValueError) as e:
        raise ValueError('Linear static solve failed. Check for rigid body modes or singularities.') from e
    u = np.zeros(n_dofs)
    u[free_dofs] = u_f
    K_g = np.zeros((n_dofs, n_dofs))
    for el_data in element_data_cache:
        (dofs, T_rot, k_e) = (el_data['dofs'], el_data['T_rot'], el_data['k_e'])
        (L, A, Iy, Iz) = (el_data['L'], el_data['A'], el_data['Iy'], el_data['Iz'])
        u_e = u[dofs]
        d_e = T_rot @ u_e
        f_e = k_e @ d_e
        P_axial = f_e[6]
        k_g = np.zeros((12, 12))
        P_30L = P_axial / (30 * L)
        mat_g_bend = P_30L * np.array([[36, 3 * L, -36, 3 * L], [3 * L, 4 * L ** 2, -3 * L, -L ** 2], [-36, -3 * L, 36, -3 * L], [3 * L, -L ** 2, -3 * L, 4 * L ** 2]])
        indices_yz = np.ix_([1, 5, 7, 11], [1, 5, 7, 11])
        k_g[indices_yz] += mat_g_bend
        mat_g_bend_w = P_30L * np.array([[36, -3 * L, -36, -3 * L], [-3 * L, 4 * L ** 2, 3 * L, -L ** 2], [-36, 3 * L, 36, 3 * L], [-3 * L, -L ** 2, 3 * L, 4 * L ** 2]])
        indices_xz = np.ix_([2, 4, 8, 10], [2, 4, 8, 10])
        k_g[indices_xz] += mat_g_bend_w
        I_p = Iy + Iz
        P_A_L_Ip = P_axial * I_p / (A * L)
        mat_g_torsion = P_A_L_Ip * np.array([[1, -1], [-1, 1]])
        indices_tx = np.ix_([3, 9], [3, 9])
        k_g[indices_tx] += mat_g_torsion
        K_g_e = T_rot.T @ k_g @ T_rot
        K_g[np.ix_(dofs, dofs)] += K_g_e
    K_g_ff = K_g[np.ix_(free_dofs, free_dofs)]
    try:
        (eigenvalues, eigenvectors) = scipy.linalg.eigh(K_ff, -K_g_ff)
    except (np.linalg.LinAlgError, ValueError) as e:
        raise ValueError('Eigenvalue solution failed. Check for singularities or ill-conditioning.') from e
    if np.any(np.iscomplex(eigenvalues)) and (not np.allclose(np.imag(eigenvalues), 0)):
        raise ValueError('Eigenvalue solution resulted in significant complex values.')
    real_eigenvalues = np.real(eigenvalues)
    positive_mask = real_eigenvalues > 1e-09
    if not np.any(positive_mask):
        raise ValueError('No positive eigenvalue found. The structure may be unstable under the reference load.')
    positive_eigenvalues = real_eigenvalues[positive_mask]
    min_pos_lambda = np.min(positive_eigenvalues)
    min_idx = np.where(real_eigenvalues == min_pos_lambda)[0][0]
    phi_f = eigenvectors[:, min_idx].real
    phi = np.zeros(n_dofs)
    phi[free_dofs] = phi_f
    return (min_pos_lambda, phi)