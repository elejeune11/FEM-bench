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

    def _build_triad_and_T(xi: np.ndarray, xj: np.ndarray, local_z: Optional[Sequence[float]]):
        d = xj - xi
        L = float(np.linalg.norm(d))
        if not np.isfinite(L) or L <= 0.0:
            raise ValueError('Element length must be positive and finite.')
        x_hat = d / L
        if local_z is None:
            ref = np.array([0.0, 0.0, 1.0], dtype=float)
            if abs(float(np.dot(x_hat, ref))) > 0.99:
                ref = np.array([0.0, 1.0, 0.0], dtype=float)
            if abs(float(np.dot(x_hat, ref))) > 0.99:
                ref = np.array([1.0, 0.0, 0.0], dtype=float)
        else:
            ref = np.asarray(local_z, dtype=float).reshape(3)
            nref = float(np.linalg.norm(ref))
            if not np.isfinite(nref) or nref == 0.0:
                ref = np.array([0.0, 0.0, 1.0], dtype=float)
        ref = ref / np.linalg.norm(ref)
        z_proj = ref - np.dot(ref, x_hat) * x_hat
        n_z = float(np.linalg.norm(z_proj))
        if n_z < 1e-12:
            alt = np.array([0.0, 1.0, 0.0], dtype=float)
            if abs(float(np.dot(x_hat, alt))) > 0.99:
                alt = np.array([1.0, 0.0, 0.0], dtype=float)
            z_proj = alt - np.dot(alt, x_hat) * x_hat
            n_z = float(np.linalg.norm(z_proj))
            if n_z < 1e-12:
                raise ValueError('Cannot construct a valid local triad for element.')
        z_hat = z_proj / n_z
        y_hat = np.cross(z_hat, x_hat)
        n_y = float(np.linalg.norm(y_hat))
        if n_y < 1e-14:
            raise ValueError('Degenerate local triad encountered.')
        y_hat = y_hat / n_y
        z_hat = np.cross(x_hat, y_hat)
        R = np.column_stack((x_hat, y_hat, z_hat))
        T = np.zeros((12, 12), dtype=float)
        Rt = R.T
        T[0:3, 0:3] = Rt
        T[3:6, 3:6] = Rt
        T[6:9, 6:9] = Rt
        T[9:12, 9:12] = Rt
        return (L, R, T)

    def _assemble_into_global(Kmat_lil, dofs, ke):
        for a in range(12):
            ia = dofs[a]
            row = ke[a, :]
            for b in range(12):
                Kmat_lil[ia, dofs[b]] += row[b]

    def local_elastic_stiffness_matrix_3D_beam(E, nu, A, L, Iy, Iz, J):
        k = np.zeros((12, 12))
        EA_L = E * A / L
        GJ_L = E * J / (2.0 * (1.0 + nu) * L)
        EIz_L = E * Iz
        EIy_L = E * Iy
        k[0, 0] = k[6, 6] = EA_L
        k[0, 6] = k[6, 0] = -EA_L
        k[3, 3] = k[9, 9] = GJ_L
        k[3, 9] = k[9, 3] = -GJ_L
        k[1, 1] = k[7, 7] = 12.0 * EIz_L / L ** 3
        k[1, 7] = k[7, 1] = -12.0 * EIz_L / L ** 3
        k[1, 5] = k[5, 1] = k[1, 11] = k[11, 1] = 6.0 * EIz_L / L ** 2
        k[5, 7] = k[7, 5] = k[7, 11] = k[11, 7] = -6.0 * EIz_L / L ** 2
        k[5, 5] = k[11, 11] = 4.0 * EIz_L / L
        k[5, 11] = k[11, 5] = 2.0 * EIz_L / L
        k[2, 2] = k[8, 8] = 12.0 * EIy_L / L ** 3
        k[2, 8] = k[8, 2] = -12.0 * EIy_L / L ** 3
        k[2, 4] = k[4, 2] = k[2, 10] = k[10, 2] = -6.0 * EIy_L / L ** 2
        k[4, 8] = k[8, 4] = k[8, 10] = k[10, 8] = 6.0 * EIy_L / L ** 2
        k[4, 4] = k[10, 10] = 4.0 * EIy_L / L
        k[4, 10] = k[10, 4] = 2.0 * EIy_L / L
        return k

    def local_geometric_stiffness_matrix_3D_beam(L: float, A: float, I_rho: float, Fx2: float, Mx2: float, My1: float, Mz1: float, My2: float, Mz2: float) -> np.ndarray:
        k_g = np.zeros((12, 12))
        k_g[0, 6] = -Fx2 / L
        k_g[1, 3] = My1 / L
        k_g[1, 4] = Mx2 / L
        k_g[1, 5] = Fx2 / 10.0
        k_g[1, 7] = -6.0 * Fx2 / (5.0 * L)
        k_g[1, 9] = My2 / L
        k_g[1, 10] = -Mx2 / L
        k_g[1, 11] = Fx2 / 10.0
        k_g[2, 3] = Mz1 / L
        k_g[2, 4] = -Fx2 / 10.0
        k_g[2, 5] = Mx2 / L
        k_g[2, 8] = -6.0 * Fx2 / (5.0 * L)
        k_g[2, 9] = Mz2 / L
        k_g[2, 10] = -Fx2 / 10.0
        k_g[2, 11] = -Mx2 / L
        k_g[3, 4] = -1.0 * (2.0 * Mz1 - Mz2) / 6.0
        k_g[3, 5] = (2.0 * My1 - My2) / 6.0
        k_g[3, 7] = -My1 / L
        k_g[3, 8] = -Mz1 / L
        k_g[3, 9] = -Fx2 * I_rho / (A * L)
        k_g[3, 10] = -1.0 * (Mz1 + Mz2) / 6.0
        k_g[3, 11] = (My1 + My2) / 6.0
        k_g[4, 7] = -Mx2 / L
        k_g[4, 8] = Fx2 / 10.0
        k_g[4, 9] = -1.0 * (Mz1 + Mz2) / 6.0
        k_g[4, 10] = -Fx2 * L / 30.0
        k_g[4, 11] = Mx2 / 2.0
        k_g[5, 7] = -Fx2 / 10.0
        k_g[5, 8] = -Mx2 / L
        k_g[5, 9] = (My1 + My2) / 6.0
        k_g[5, 10] = -Mx2 / 2.0
        k_g[5, 11] = -Fx2 * L / 30.0
        k_g[7, 9] = -My2 / L
        k_g[7, 10] = Mx2 / L
        k_g[7, 11] = -Fx2 / 10.0
        k_g[8, 9] = -Mz2 / L
        k_g[8, 10] = Fx2 / 10.0
        k_g[8, 11] = Mx2 / L
        k_g[9, 10] = (Mz1 - 2.0 * Mz2) / 6.0
        k_g[9, 11] = -1.0 * (My1 - 2.0 * My2) / 6.0
        k_g = k_g + k_g.transpose()
        k_g[0, 0] = Fx2 / L
        k_g[1, 1] = 6.0 * Fx2 / (5.0 * L)
        k_g[2, 2] = 6.0 * Fx2 / (5.0 * L)
        k_g[3, 3] = Fx2 * I_rho / (A * L)
        k_g[4, 4] = 2.0 * Fx2 * L / 15.0
        k_g[5, 5] = 2.0 * Fx2 * L / 15.0
        k_g[6, 6] = Fx2 / L
        k_g[7, 7] = 6.0 * Fx2 / (5.0 * L)
        k_g[8, 8] = 6.0 * Fx2 / (5.0 * L)
        k_g[9, 9] = Fx2 * I_rho / (A * L)
        k_g[10, 10] = 2.0 * Fx2 * L / 15.0
        k_g[11, 11] = 2.0 * Fx2 * L / 15.0
        return k_g
    coords = np.asarray(node_coords, dtype=float)
    if coords.ndim != 2 or coords.shape[1] != 3:
        raise ValueError('node_coords must be an (n_nodes, 3) array')
    n_nodes = coords.shape[0]
    n_dof = 6 * n_nodes
    K = scipy.sparse.lil_matrix((n_dof, n_dof), dtype=float)
    element_data = []
    for e in elements:
        try:
            ni = int(e['node_i'])
            nj = int(e['node_j'])
            E = float(e['E'])
            nu = float(e['nu'])
            A = float(e['A'])
            Iy = float(e['Iy'])
            Iz = float(e['Iz'])
            J = float(e['J'])
            I_rho = float(e['I_rho'])
            local_z = e.get('local_z', None)
        except Exception as ex:
            raise ValueError(f'Invalid element definition: {ex}')
        if not (0 <= ni < n_nodes and 0 <= nj < n_nodes):
            raise ValueError('Element node indices out of range.')
        xi = coords[ni]
        xj = coords[nj]
        (L, R, T) = _build_triad_and_T(xi, xj, local_z)
        k_loc = local_elastic_stiffness_matrix_3D_beam(E, nu, A, L, Iy, Iz, J)
        k_glob = T.T @ k_loc @ T
        dofs = np.concatenate((np.arange(6 * ni, 6 * ni + 6, dtype=int), np.arange(6 * nj, 6 * nj + 6, dtype=int)))
        _assemble_into_global(K, dofs, k_glob)
        element_data.append({'dofs': dofs, 'T': T, 'k_loc': k_loc, 'L': L, 'A': A, 'I_rho': I_rho})
    P = np.zeros(n_dof, dtype=float)
    for (node_idx, loads) in nodal_loads.items():
        base = 6 * int(node_idx)
        if not 0 <= base < n_dof:
            raise ValueError('Nodal load node index out of range.')
        lv = np.asarray(loads, dtype=float).reshape(-1)
        if lv.size != 6:
            raise ValueError('Each nodal load vector must have length 6.')
        P[base:base + 6] += lv
    constrained = np.zeros(n_dof, dtype=bool)
    for (node_idx, spec) in boundary_conditions.items():
        base = 6 * int(node_idx)
        if not 0 <= base < n_dof:
            raise ValueError('Boundary condition node index out of range.')
        sp = list(spec)
        use_bool = False
        if len(sp) == 6 and all((isinstance(x, (bool, np.bool_)) for x in sp)):
            use_bool = True
        if use_bool:
            mask = np.array(sp, dtype=bool)
            constrained[base:base + 6] = mask | constrained[base:base + 6]
        else:
            for d in sp:
                di = int(d)
                if not 0 <= di < 6:
                    raise ValueError('Boundary condition DOF index out of range.')
                constrained[base + di] = True
    free = ~constrained
    if not np.any(free):
        raise ValueError('No free DOFs remain after applying boundary conditions.')
    K_csr = K.tocsr()
    K_ff = K_csr[free, :][:, free]
    P_f = P[free]
    try:
        u_f = scipy.sparse.linalg.spsolve(K_ff, P_f)
    except Exception as ex:
        raise ValueError(f'Linear solve failed; stiffness matrix may be singular: {ex}')
    if not np.all(np.isfinite(u_f)):
        raise ValueError('Non-finite values in displacement solution.')
    u = np.zeros(n_dof, dtype=float)
    u[free] = u_f
    K_g = scipy.sparse.lil_matrix((n_dof, n_dof), dtype=float)
    any_nonzero_kg = False
    for ed in element_data:
        dofs = ed['dofs']
        T = ed['T']
        k_loc = ed['k_loc']
        L = ed['L']
        A = ed['A']
        I_rho = ed['I_rho']
        u_e_glob = u[dofs]
        u_loc = T @ u_e_glob
        f_loc = k_loc @ u_loc
        Fx2 = float(f_loc[6])
        Mx2 = float(f_loc[9])
        My1 = float(f_loc[4])
        Mz1 = float(f_loc[5])
        My2 = float(f_loc[10])
        Mz2 = float(f_loc[11])
        k_g_loc = local_geometric_stiffness_matrix_3D_beam(L, A, I_rho, Fx2, Mx2, My1, Mz1, My2, Mz2)
        if not np.allclose(k_g_loc, 0.0):
            any_nonzero_kg = True
        k_g_glob = T.T @ k_g_loc @ T
        _assemble_into_global(K_g, dofs, k_g_glob)
    if not any_nonzero_kg:
        raise ValueError('Geometric stiffness is zero; no buckling mode can be determined for the given reference state.')
    K_g_csr = K_g.tocsr()
    K_g_ff = K_g_csr[free, :][:, free]
    A = K_ff.toarray()
    B = (-K_g_ff).toarray()
    try:
        (w, v) = scipy.linalg.eig(A, B, check_finite=False)
    except Exception as ex:
        raise ValueError(f'Eigenvalue solve failed: {ex}')
    if w.size == 0:
        raise ValueError('No eigenvalues returned by solver.')
    wr = np.real(w)
    wi = np.imag(w)
    tol_im = 1e-08
    real_mask = np.abs(wi) <= tol_im * np.maximum(1.0, np.abs(wr))
    pos_mask = wr > max(1e-12, np.finfo(float).eps * 10)
    candidates = np.where(real_mask & pos_mask)[0]
    if candidates.size == 0:
        raise ValueError('No positive real eigenvalue found.')
    idx = candidates[np.argmin(wr[candidates])]
    lam = float(wr[idx])
    phi_f = np.real(v[:, idx])
    if not np.any(np.isfinite(phi_f)):
        raise ValueError('Invalid eigenvector computed.')
    phi = np.zeros(n_dof, dtype=float)
    phi[free] = phi_f
    return (lam, phi)