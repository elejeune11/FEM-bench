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
    n_nodes = int(node_coords.shape[0])
    if node_coords.shape[1] != 3:
        raise ValueError('node_coords must be of shape (n_nodes, 3)')
    n_dof = 6 * n_nodes
    K = np.zeros((n_dof, n_dof), dtype=float)
    Kg = np.zeros((n_dof, n_dof), dtype=float)
    P = np.zeros(n_dof, dtype=float)
    for (n, load) in (nodal_loads or {}).items():
        idx = 6 * int(n)
        arr = np.asarray(load, dtype=float).reshape(-1)
        if arr.size != 6:
            raise ValueError('Each nodal load must be length 6')
        P[idx:idx + 6] += arr

    def element_rotation_and_T(i_node, j_node, local_z_vec):
        xi = node_coords[i_node].astype(float)
        xj = node_coords[j_node].astype(float)
        dx = xj - xi
        L = float(np.linalg.norm(dx))
        if not np.isfinite(L) or L <= 0.0:
            raise ValueError('Element length must be positive')
        ex = dx / L
        ez = None
        if local_z_vec is not None:
            v = np.asarray(local_z_vec, dtype=float).reshape(3)
            if not np.all(np.isfinite(v)):
                v = None
            else:
                proj = v - np.dot(v, ex) * ex
                nrm = np.linalg.norm(proj)
                if nrm > 0.0:
                    ez = proj / nrm
        if ez is None:
            g = np.array([0.0, 0.0, 1.0])
            if abs(np.dot(ex, g)) > 1.0 - 1e-08:
                g = np.array([0.0, 1.0, 0.0])
            c = np.cross(ex, g)
            nc = np.linalg.norm(c)
            if nc <= 0.0:
                raise ValueError('Failed to construct local triad')
            ez = c / nc
        ey = np.cross(ez, ex)
        ey /= np.linalg.norm(ey)
        R = np.column_stack((ex, ey, ez))
        T = np.zeros((12, 12), dtype=float)
        Rt = R.T
        T[0:3, 0:3] = Rt
        T[3:6, 3:6] = Rt
        T[6:9, 6:9] = Rt
        T[9:12, 9:12] = Rt
        return (L, R, T)
    for el in elements:
        i = int(el['node_i'])
        j = int(el['node_j'])
        E = float(el['E'])
        nu = float(el['nu'])
        A = float(el['A'])
        Iy = float(el['Iy'])
        Iz = float(el['Iz'])
        J = float(el['J'])
        local_z = el.get('local_z', None)
        (L, R, T) = element_rotation_and_T(i, j, local_z)
        ke_local = local_elastic_stiffness_matrix_3D_beam(E, nu, A, L, Iy, Iz, J)
        ke_global = T.T @ ke_local @ T
        dofs = np.r_[np.arange(6 * i, 6 * i + 6), np.arange(6 * j, 6 * j + 6)]
        K[np.ix_(dofs, dofs)] += ke_global
    K = 0.5 * (K + K.T)
    constrained = np.zeros(n_dof, dtype=bool)
    if boundary_conditions is not None:
        for (n, spec) in boundary_conditions.items():
            base = 6 * int(n)
            s = list(spec)
            if len(s) == 6 and all((isinstance(x, (bool, np.bool_)) for x in s)):
                for k in range(6):
                    if bool(s[k]):
                        constrained[base + k] = True
            else:
                for d in s:
                    di = int(d)
                    if di < 0 or di >= 6:
                        raise ValueError('Invalid DOF index in boundary condition')
                    constrained[base + di] = True
    free = np.where(~constrained)[0]
    if free.size == 0:
        raise ValueError('No free DOFs remain after applying boundary conditions')
    Kff = K[np.ix_(free, free)]
    Pf = P[free]
    try:
        (c, lower) = scipy.linalg.cho_factor(Kff, lower=True, check_finite=False, overwrite_a=False)
        uf = scipy.linalg.cho_solve((c, lower), Pf, check_finite=False, overwrite_b=False)
    except Exception as e:
        raise ValueError('Linear system K_ff u = P_f failed (matrix not SPD or singular)') from e
    u = np.zeros(n_dof, dtype=float)
    u[free] = uf
    for el in elements:
        i = int(el['node_i'])
        j = int(el['node_j'])
        E = float(el['E'])
        nu = float(el['nu'])
        A = float(el['A'])
        Iy = float(el['Iy'])
        Iz = float(el['Iz'])
        J = float(el['J'])
        I_rho = float(el['I_rho'])
        local_z = el.get('local_z', None)
        (L, R, T) = element_rotation_and_T(i, j, local_z)
        ke_local = local_elastic_stiffness_matrix_3D_beam(E, nu, A, L, Iy, Iz, J)
        dofs = np.r_[np.arange(6 * i, 6 * i + 6), np.arange(6 * j, 6 * j + 6)]
        d_global_e = u[dofs]
        d_local_e = T @ d_global_e
        f_local_e = ke_local @ d_local_e
        Fx2 = float(f_local_e[6])
        Mx2 = float(f_local_e[9])
        My1 = float(f_local_e[4])
        Mz1 = float(f_local_e[5])
        My2 = float(f_local_e[10])
        Mz2 = float(f_local_e[11])
        kg_local = local_geometric_stiffness_matrix_3D_beam(L, A, I_rho, Fx2, Mx2, My1, Mz1, My2, Mz2)
        kg_global = T.T @ kg_local @ T
        Kg[np.ix_(dofs, dofs)] += kg_global
    Kg = 0.5 * (Kg + Kg.T)
    Kgff = Kg[np.ix_(free, free)]
    if np.linalg.norm(Kgff, ord='fro') <= 1e-14 * max(1.0, np.linalg.norm(Kff, ord='fro')):
        raise ValueError('Geometric stiffness is (near) zero under the reference load; cannot perform buckling analysis.')
    try:
        (mu, vecs) = scipy.linalg.eigh(Kgff, Kff, check_finite=False)
    except Exception as e:
        raise ValueError('Generalized eigenvalue solve failed') from e
    abs_mu_max = float(np.max(np.abs(mu))) if mu.size > 0 else 0.0
    tol_mu = 1e-12 * max(1.0, abs_mu_max)
    mask = mu < -tol_mu
    if not np.any(mask):
        raise ValueError('No positive buckling factor found (no negative eigenvalues of Kgff vs Kff)')
    lambdas = -1.0 / mu[mask]
    pos_mask = np.isfinite(lambdas) & (lambdas > 0.0)
    if not np.any(pos_mask):
        raise ValueError('No positive finite buckling factors found')
    idx_local = np.argmin(lambdas[pos_mask])
    candidate_indices = np.where(mask)[0][pos_mask]
    eig_index = candidate_indices[idx_local]
    phi_f = vecs[:, eig_index]
    phi = np.zeros(n_dof, dtype=float)
    phi[free] = phi_f
    return (float(lambdas[pos_mask][idx_local]), phi)