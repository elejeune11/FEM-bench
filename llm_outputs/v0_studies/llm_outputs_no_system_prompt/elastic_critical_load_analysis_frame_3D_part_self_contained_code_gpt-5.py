def elastic_critical_load_analysis_frame_3D_part_self_contained(node_coords: np.ndarray, elements: Sequence[dict], boundary_conditions: dict[int, Sequence[int | bool]], nodal_loads: dict[int, Sequence[float]]):
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
    node_coords : (n_nodes, 3) ndarray of float
        Cartesian coordinates (x, y, z) for each node, indexed 0..n_nodes-1.
    elements : Sequence[dict]
        Element definitions consumed by the assembly routines. Each dictionary
        must supply properties for a 2-node 3D Euler-Bernoulli beam aligned with
        its local x-axis. Required keys (minimum):
          Topology
          --------
                Start node index (0-based).
                End node index (0-based).
          Material
          --------
                Young's modulus (used in axial, bending, and torsion terms).
                Poisson's ratio (used in torsion only).
          Section (local axes y,z about the beam's local x)
          -----------------------------------------------
                Cross-sectional area.
                Second moment of area about local y.
                Second moment of area about local z.
                Torsional constant (for elastic/torsional stiffness).
                Polar moment about the local x-axis used by the geometric stiffness
                with torsion–bending coupling.
          Orientation
          -----------
                Provide a 3-vector giving the direction of the element's local z-axis to 
                disambiguate the local triad used in the 12×12 transformation; if `None`, 
                a default convention is applied.
    boundary_conditions : dict
        Dictionary mapping node index -> boundary condition specification. Each
        node's specification can be provided in either of two forms:
          the DOF is constrained (fixed).
          at that node are fixed.
        All constrained DOFs are removed from the free set. It is the caller’s
        responsibility to supply constraints sufficient to eliminate rigid-body
        modes.
    nodal_loads : dict[int, Sequence[float]]
        Mapping from node index → length-6 vector of load components applied at
        that node in the **global** DOF order `[F_x, F_y, F_z, M_x, M_y, M_z]`.
        Used to form `P`.
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
    External Helper Functions (required)
    ------------------------------------
        Local elastic stiffness matrix for a 3D Euler-Bernoulli beam aligned with
        the local x-axis.
        Local geometric stiffness matrix with torsion-bending coupling.
    Raises
    ------
    ValueError
        Propagated from called routines if:
    Notes
    -----
      if desired (e.g., by max absolute translational DOF).
      the returned mode can depend on numerical details.
    """
    if not isinstance(node_coords, np.ndarray) or node_coords.ndim != 2 or node_coords.shape[1] != 3:
        raise ValueError('node_coords must be an (n_nodes, 3) ndarray')
    n_nodes = node_coords.shape[0]
    n_dof = 6 * n_nodes
    if n_nodes < 2:
        raise ValueError('At least two nodes are required')

    def build_rotation_and_T(xi: np.ndarray, xj: np.ndarray, local_z_hint: Optional[Sequence[float]]):
        dx = xj - xi
        L = float(np.linalg.norm(dx))
        if not np.isfinite(L) or L <= 0.0:
            raise ValueError('Element length must be positive and finite')
        ex = dx / L
        if local_z_hint is not None:
            z_ref = np.asarray(local_z_hint, dtype=float).reshape(3)
        else:
            z_ref = np.array([0.0, 0.0, 1.0], dtype=float)
        if np.linalg.norm(np.cross(z_ref, ex)) < 1e-08:
            alt = np.array([0.0, 1.0, 0.0], dtype=float)
            if np.linalg.norm(np.cross(alt, ex)) < 1e-08:
                alt = np.array([1.0, 0.0, 0.0], dtype=float)
            z_ref = alt
        ey_unnorm = np.cross(z_ref, ex)
        ey_norm = np.linalg.norm(ey_unnorm)
        if ey_norm < 1e-12:
            if abs(ex[2]) < 0.9:
                z_ref = np.array([0.0, 0.0, 1.0], dtype=float)
            else:
                z_ref = np.array([0.0, 1.0, 0.0], dtype=float)
            ey_unnorm = np.cross(z_ref, ex)
            ey_norm = np.linalg.norm(ey_unnorm)
        ey = ey_unnorm / ey_norm
        ez = np.cross(ex, ey)
        ez = ez / np.linalg.norm(ez)
        ey = np.cross(ez, ex)
        ey = ey / np.linalg.norm(ey)
        R = np.vstack((ex, ey, ez))
        T = np.zeros((12, 12), dtype=float)
        T[0:3, 0:3] = R
        T[3:6, 3:6] = R
        T[6:9, 6:9] = R
        T[9:12, 9:12] = R
        return (L, R, T)
    constrained = np.zeros(n_dof, dtype=bool)
    for (node_idx, spec) in boundary_conditions.items():
        if node_idx < 0 or node_idx >= n_nodes:
            raise ValueError('Boundary condition specified for invalid node index')
        base = 6 * node_idx
        spec_seq = list(spec)
        if len(spec_seq) == 6 and all((isinstance(x, (bool, np.bool_)) for x in spec_seq)):
            for (k, flag) in enumerate(spec_seq):
                if flag:
                    constrained[base + k] = True
        else:
            for dof_local in spec_seq:
                if not isinstance(dof_local, (int, np.integer)):
                    raise ValueError('Boundary condition DOF indices must be integers or provide 6 booleans')
                if dof_local < 0 or dof_local >= 6:
                    raise ValueError('Boundary condition DOF indices must be in [0..5]')
                constrained[base + int(dof_local)] = True
    free = np.where(~constrained)[0]
    if free.size == 0:
        raise ValueError('All DOFs are constrained; no free DOFs to analyze')
    K = np.zeros((n_dof, n_dof), dtype=float)
    P = np.zeros(n_dof, dtype=float)
    for (nid, load) in nodal_loads.items():
        if nid < 0 or nid >= n_nodes:
            raise ValueError('Load specified for invalid node index')
        load_vec = np.asarray(load, dtype=float).reshape(-1)
        if load_vec.size != 6:
            raise ValueError('Each nodal load must be a length-6 sequence')
        P[6 * nid:6 * nid + 6] += load_vec
    elem_data = []
    for e in elements:
        ni = int(e['node_i'])
        nj = int(e['node_j'])
        if ni < 0 or nj < 0 or ni >= n_nodes or (nj >= n_nodes):
            raise ValueError('Element references invalid node indices')
        xi = node_coords[ni].astype(float)
        xj = node_coords[nj].astype(float)
        local_z_hint = e.get('local_z', None)
        E = float(e['E'])
        nu = float(e['nu'])
        A = float(e['A'])
        Iy = float(e['Iy'])
        Iz = float(e['Iz'])
        J = float(e['J'])
        I_rho = float(e['I_rho'])
        (L, R, T) = build_rotation_and_T(xi, xj, local_z_hint)
        k_loc = local_elastic_stiffness_matrix_3D_beam(E, nu, A, L, Iy, Iz, J)
        k_glob = T.T @ k_loc @ T
        dofs = np.array([6 * ni + 0, 6 * ni + 1, 6 * ni + 2, 6 * ni + 3, 6 * ni + 4, 6 * ni + 5, 6 * nj + 0, 6 * nj + 1, 6 * nj + 2, 6 * nj + 3, 6 * nj + 4, 6 * nj + 5], dtype=int)
        K[np.ix_(dofs, dofs)] += k_glob
        elem_data.append({'nodes': (ni, nj), 'L': L, 'A': A, 'I_rho': I_rho, 'T': T, 'k_loc': k_loc, 'dofs': dofs})
    K_ff = K[np.ix_(free, free)]
    P_f = P[free]
    if not np.all(np.isfinite(K_ff)) or not np.all(np.isfinite(P_f)):
        raise ValueError('Non-finite entries encountered in system matrices')
    try:
        condK = np.linalg.cond(K_ff)
    except np.linalg.LinAlgError:
        condK = np.inf
    if not np.isfinite(condK) or condK > 1000000000000.0:
        raise ValueError('Reduced elastic stiffness matrix is singular or ill-conditioned')
    try:
        u_f = np.linalg.solve(K_ff, P_f)
    except np.linalg.LinAlgError as err:
        raise ValueError(f'Failed to solve static problem on free DOFs: {err}')
    u = np.zeros(n_dof, dtype=float)
    u[free] = u_f
    K_g = np.zeros((n_dof, n_dof), dtype=float)
    total_force_scale = 0.0
    for data in elem_data:
        dofs = data['dofs']
        T = data['T']
        k_loc = data['k_loc']
        L = data['L']
        A = data['A']
        I_rho = data['I_rho']
        u_e_glob = u[dofs]
        d_loc = T @ u_e_glob
        f_loc = k_loc @ d_loc
        if not np.all(np.isfinite(f_loc)):
            raise ValueError('Non-finite internal force resultants computed')
        Fx2 = float(f_loc[6])
        Mx2 = float(f_loc[9])
        My1 = float(f_loc[4])
        Mz1 = float(f_loc[5])
        My2 = float(f_loc[10])
        Mz2 = float(f_loc[11])
        total_force_scale += abs(Fx2) + abs(My1) + abs(Mz1) + abs(My2) + abs(Mz2) + abs(Mx2)
        k_g_loc = local_geometric_stiffness_matrix_3D_beam(L, A, I_rho, Fx2, Mx2, My1, Mz1, My2, Mz2)
        k_g_glob = T.T @ k_g_loc @ T
        K_g[np.ix_(dofs, dofs)] += k_g_glob
    if total_force_scale == 0.0 or not np.any(np.abs(K_g) > 0):
        raise ValueError('Geometric stiffness is zero under the provided reference load; cannot perform buckling analysis')
    K_ff = 0.5 * (K_ff + K_ff.T)
    K_g_ff = K_g[np.ix_(free, free)]
    K_g_ff = 0.5 * (K_g_ff + K_g_ff.T)
    B_ff = -K_g_ff
    if not np.all(np.isfinite(B_ff)):
        raise ValueError('Non-finite entries in geometric stiffness')
    try:
        (w, vecs) = scipy.linalg.eigh(a=K_ff, b=B_ff, turbo=True, check_finite=True)
    except Exception as err:
        raise ValueError(f'Generalized eigenvalue solve failed: {err}')
    tol = 1e-10
    valid_idx = np.where(np.isfinite(w) & (w > tol))[0]
    if valid_idx.size == 0:
        raise ValueError('No positive eigenvalues found for buckling problem')
    i_min = valid_idx[np.argmin(w[valid_idx])]
    lambda_cr = float(w[i_min])
    phi_f = vecs[:, i_min]
    phi = np.zeros(n_dof, dtype=float)
    phi[free] = phi_f
    return (lambda_cr, phi)