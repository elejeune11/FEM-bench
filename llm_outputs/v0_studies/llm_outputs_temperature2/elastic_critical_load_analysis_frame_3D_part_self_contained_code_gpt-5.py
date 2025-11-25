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

    def _normalize(v):
        n = np.linalg.norm(v)
        if n <= 0.0:
            return v
        return v / n

    def _element_rotation_and_T(xi, xj, local_z_vec):
        dx = xj - xi
        L = float(np.linalg.norm(dx))
        if not np.isfinite(L) or L <= 0.0:
            raise ValueError('Element has zero or invalid length.')
        ex = dx / L
        ez = None
        if local_z_vec is not None:
            zref = np.asarray(local_z_vec, dtype=float).reshape(3)
            if np.linalg.norm(zref) > 0.0:
                ztmp = zref - np.dot(zref, ex) * ex
                nt = np.linalg.norm(ztmp)
                if nt > 1e-14:
                    ez = ztmp / nt
        if ez is None:
            up = np.array([0.0, 0.0, 1.0])
            if np.linalg.norm(np.cross(ex, up)) < 1e-08:
                up = np.array([0.0, 1.0, 0.0])
            ez = _normalize(np.cross(ex, up))
        ey = _normalize(np.cross(ez, ex))
        R = np.column_stack((ex, ey, ez))
        T = np.zeros((12, 12))
        T[0:3, 0:3] = R
        T[3:6, 3:6] = R
        T[6:9, 6:9] = R
        T[9:12, 9:12] = R
        return (L, R, T)

    def _global_dof_indices(i, j):
        bi = 6 * int(i)
        bj = 6 * int(j)
        return list(range(bi, bi + 6)) + list(range(bj, bj + 6))
    n_nodes = int(node_coords.shape[0])
    n_dof = 6 * n_nodes
    K = np.zeros((n_dof, n_dof), dtype=float)
    P = np.zeros(n_dof, dtype=float)
    for (n, load) in (nodal_loads or {}).items():
        idx = 6 * int(n)
        f = np.asarray(load, dtype=float).reshape(-1)
        if f.size != 6:
            raise ValueError('Each nodal load must be length 6.')
        P[idx:idx + 6] += f
    for e in elements:
        i = int(e['node_i'])
        j = int(e['node_j'])
        xi = np.asarray(node_coords[i], dtype=float)
        xj = np.asarray(node_coords[j], dtype=float)
        (L, R, T) = _element_rotation_and_T(xi, xj, e.get('local_z', None))
        E = float(e['E'])
        nu = float(e['nu'])
        A = float(e['A'])
        Iy = float(e['Iy'])
        Iz = float(e['Iz'])
        J = float(e['J'])
        k_loc = local_elastic_stiffness_matrix_3D_beam(E, nu, A, L, Iy, Iz, J)
        k_glob = T @ k_loc @ T.T
        dofs = _global_dof_indices(i, j)
        K[np.ix_(dofs, dofs)] += k_glob
    constrained = set()
    if boundary_conditions is not None:
        for (n, spec) in boundary_conditions.items():
            base = 6 * int(n)
            seq = list(spec)
            if len(seq) == 6 and all((isinstance(x, bool) for x in seq)):
                inds = [k for (k, b) in enumerate(seq) if b]
            else:
                inds = [int(k) for k in seq]
            for d in inds:
                if d < 0 or d > 5:
                    raise ValueError('Invalid DOF index in boundary conditions.')
                constrained.add(base + d)
    all_dofs = set(range(n_dof))
    free = sorted(all_dofs - constrained)
    if len(free) == 0:
        raise ValueError('No free DOFs remain after applying boundary conditions.')
    constrained = sorted(constrained)
    K_ff = K[np.ix_(free, free)]
    P_f = P[free]
    try:
        u_f = scipy.linalg.solve(K_ff, P_f, assume_a='sym')
    except Exception as err:
        raise ValueError(f'Static solve failed: {err}') from err
    u = np.zeros(n_dof, dtype=float)
    u[free] = u_f
    K_g = np.zeros((n_dof, n_dof), dtype=float)
    for e in elements:
        i = int(e['node_i'])
        j = int(e['node_j'])
        xi = np.asarray(node_coords[i], dtype=float)
        xj = np.asarray(node_coords[j], dtype=float)
        (L, R, T) = _element_rotation_and_T(xi, xj, e.get('local_z', None))
        E = float(e['E'])
        nu = float(e['nu'])
        A = float(e['A'])
        Iy = float(e['Iy'])
        Iz = float(e['Iz'])
        J = float(e['J'])
        I_rho = float(e['I_rho'])
        k_loc = local_elastic_stiffness_matrix_3D_beam(E, nu, A, L, Iy, Iz, J)
        dofs = _global_dof_indices(i, j)
        u_e_global = u[dofs]
        u_e_local = T.T @ u_e_global
        f_e_local = k_loc @ u_e_local
        Fx2 = float(f_e_local[6])
        Mx2 = float(f_e_local[9])
        My1 = float(f_e_local[4])
        Mz1 = float(f_e_local[5])
        My2 = float(f_e_local[10])
        Mz2 = float(f_e_local[11])
        k_g_loc = local_geometric_stiffness_matrix_3D_beam(L=L, A=A, I_rho=I_rho, Fx2=Fx2, Mx2=Mx2, My1=My1, Mz1=Mz1, My2=My2, Mz2=Mz2)
        k_g_glob = T @ k_g_loc @ T.T
        K_g[np.ix_(dofs, dofs)] += k_g_glob
    Kg_ff = K_g[np.ix_(free, free)]
    K_ff = 0.5 * (K_ff + K_ff.T)
    Kg_ff = 0.5 * (Kg_ff + Kg_ff.T)
    B = -Kg_ff
    try:
        (eigvals, eigvecs) = scipy.linalg.eig(K_ff, B)
    except Exception as err:
        raise ValueError(f'Generalized eigenvalue solve failed: {err}') from err
    eigvals = np.asarray(eigvals)
    eigvecs = np.asarray(eigvecs)
    if eigvals.size == 0:
        raise ValueError('No eigenvalues returned by solver.')
    real_parts = eigvals.real
    imag_parts = eigvals.imag
    tol_imag = 1e-08
    finite_mask = np.isfinite(real_parts) & np.isfinite(imag_parts)
    real_mask = np.abs(imag_parts) <= np.maximum(1.0, np.abs(real_parts)) * tol_imag
    pos_mask = real_parts > 0.0
    mask = finite_mask & real_mask & pos_mask
    if not np.any(mask):
        raise ValueError('No positive real eigenvalue found for buckling.')
    idx_candidates = np.where(mask)[0]
    idx_min = idx_candidates[np.argmin(real_parts[idx_candidates])]
    lam = float(real_parts[idx_min])
    phi_free = eigvecs[:, idx_min].real
    phi = np.zeros(n_dof, dtype=float)
    phi[free] = phi_free
    return (lam, phi)