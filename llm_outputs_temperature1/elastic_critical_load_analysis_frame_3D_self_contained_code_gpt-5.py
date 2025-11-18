def elastic_critical_load_analysis_frame_3D_self_contained(node_coords: np.ndarray, elements: Sequence[dict], boundary_conditions: dict[int, Sequence[int | bool]], nodal_loads: dict[int, Sequence[float]]):
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
                Poisson's ratio (used in torsion only, per your stiffness routine).
          Section (local axes y,z about the beam's local x)
          -----------------------------------------------
                Cross-sectional area.
                Second moment of area about local y.
                Second moment of area about local z.
                Torsional constant (for elastic/torsional stiffness).
                Polar moment about the local x-axis used by the geometric stiffness
                with torsion-bending coupling (see your geometric K routine).
          Orientation
          -----------
                Provide a 3-vector giving the direction of the element's local z-axis to 
                disambiguate the local triad used in the 12x12 transformation; if set to `None`, 
                a default convention will be applied to construct the local axes.
    boundary_conditions : dict
        Dictionary mapping node index -> boundary condition specification. Each
        node’s specification can be provided in either of two forms:
          the DOF is constrained (fixed).
          at that node are fixed.
        All constrained DOFs are removed from the free set. It is the caller’s
        responsibility to supply constraints sufficient to eliminate rigid-body
        modes.
    nodal_loads : dict[int, Sequence[float]]
        Mapping from node index → length-6 vector of load components applied at
        that node in the **global** DOF order `[F_x, F_y, F_z, M_x, M_y, M_z]`.
        Consumed by `assemble_global_load_vector_linear_elastic_3D` to form `P`.
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
    """

    def dof_index(nid: int, comp: int) -> int:
        return nid * 6 + comp
    n_nodes = int(node_coords.shape[0])
    ndof = n_nodes * 6
    constrained = np.zeros(ndof, dtype=bool)
    if boundary_conditions is not None:
        for (nid, spec) in boundary_conditions.items():
            if spec is None:
                continue
            try:
                seq = list(spec)
            except TypeError:
                raise ValueError('Boundary condition specification must be a sequence')
            if len(seq) == 6 and all((isinstance(x, (bool, np.bool_)) for x in seq)):
                for k in range(6):
                    if bool(seq[k]):
                        constrained[dof_index(nid, k)] = True
            else:
                for idx in seq:
                    ii = int(idx)
                    if ii < 0 or ii > 5:
                        raise ValueError('Boundary condition DOF index must be in 0..5')
                    constrained[dof_index(nid, ii)] = True
    free_dofs = np.where(~constrained)[0]
    if free_dofs.size == 0:
        raise ValueError('All DOFs constrained; no free DOFs to analyze')
    K = np.zeros((ndof, ndof), dtype=float)

    def normalize(v: np.ndarray) -> np.ndarray:
        n = np.linalg.norm(v)
        if n <= 0:
            raise ValueError('Zero-length vector encountered during axis construction')
        return v / n

    def build_rotation_matrix(i: int, j: int, local_z_guess: Optional[Sequence[float]]) -> np.ndarray:
        xi = node_coords[i]
        xj = node_coords[j]
        dx = xj - xi
        L = float(np.linalg.norm(dx))
        if L <= 0:
            raise ValueError('Zero-length element detected')
        ex = dx / L
        if local_z_guess is not None:
            zg = np.array(local_z_guess, dtype=float).reshape(3)
        else:
            zg = np.array([0.0, 0.0, 1.0], dtype=float)
        if abs(np.dot(ex, normalize(zg))) > 0.999:
            zg = np.array([0.0, 1.0, 0.0], dtype=float)
        ey = np.cross(zg, ex)
        if np.linalg.norm(ey) < 1e-12:
            zg = np.array([0.0, 1.0, 0.0], dtype=float)
            ey = np.cross(zg, ex)
        ey = normalize(ey)
        ez = np.cross(ex, ey)
        ez = normalize(ez)
        R = np.zeros((3, 3), dtype=float)
        R[0, :] = ex
        R[1, :] = ey
        R[2, :] = ez
        return R

    def build_T(R: np.ndarray) -> np.ndarray:
        T = np.zeros((12, 12), dtype=float)
        T[:3, :3] = R
        T[3:6, 3:6] = R
        T[6:9, 6:9] = R
        T[9:12, 9:12] = R
        return T

    def elastic_stiffness_local(E: float, nu: float, A: float, Iy: float, Iz: float, J: float, L: float) -> np.ndarray:
        k = np.zeros((12, 12), dtype=float)
        L2 = L * L
        L3 = L2 * L
        EA_L = E * A / L
        k[0, 0] += EA_L
        k[0, 6] += -EA_L
        k[6, 0] += -EA_L
        k[6, 6] += EA_L
        G = E / (2.0 * (1.0 + nu))
        GJ_L = G * J / L
        k[3, 3] += GJ_L
        k[3, 9] += -GJ_L
        k[9, 3] += -GJ_L
        k[9, 9] += GJ_L
        EIz = E * Iz
        c = EIz / L3
        idx_bz = [1, 5, 7, 11]
        M = np.array([[12.0, 6.0 * L, -12.0, 6.0 * L], [6.0 * L, 4.0 * L2, -6.0 * L, 2.0 * L2], [-12.0, -6.0 * L, 12.0, -6.0 * L], [6.0 * L, 2.0 * L2, -6.0 * L, 4.0 * L2]], dtype=float)
        kbz = c * M
        for a in range(4):
            for b in range(4):
                k[idx_bz[a], idx_bz[b]] += kbz[a, b]
        EIy = E * Iy
        c = EIy / L3
        idx_by = [2, 4, 8, 10]
        kby = c * M
        for a in range(4):
            for b in range(4):
                k[idx_by[a], idx_by[b]] += kby[a, b]
        return k

    def geometric_stiffness_local(N_comp: float, L: float) -> np.ndarray:
        kg = np.zeros((12, 12), dtype=float)
        if abs(N_comp) <= 0.0:
            return kg
        L2 = L * L
        Mg = np.array([[36.0, 3.0 * L, -36.0, 3.0 * L], [3.0 * L, 4.0 * L2, -3.0 * L, -1.0 * L2], [-36.0, -3.0 * L, 36.0, -3.0 * L], [3.0 * L, -1.0 * L2, -3.0 * L, 4.0 * L2]], dtype=float)
        coeff = -N_comp / (30.0 * L)
        Kg_block = coeff * Mg
        idx_bz = [1, 5, 7, 11]
        idx_by = [2, 4, 8, 10]
        for a in range(4):
            ia_bz = idx_bz[a]
            ia_by = idx_by[a]
            for b in range(4):
                kg[ia_bz, idx_bz[b]] += Kg_block[a, b]
                kg[ia_by, idx_by[b]] += Kg_block[a, b]
        return kg
    P = np.zeros(ndof, dtype=float)
    if nodal_loads is not None:
        for (nid, load) in nodal_loads.items():
            vec = np.array(load, dtype=float).reshape(6)
            base = nid * 6
            P[base:base + 6] += vec
    elem_data = []
    for el in elements:
        i = int(el['node_i'])
        j = int(el['node_j'])
        E = float(el['E'])
        nu = float(el['nu'])
        A = float(el['A'])
        Iy = float(el['I_y'])
        Iz = float(el['I_z'])
        J = float(el['J'])
        _ = el.get('I_rho', None)
        local_z = el.get('local_z', None)
        R = build_rotation_matrix(i, j, local_z)
        xi = node_coords[i]
        xj = node_coords[j]
        L = float(np.linalg.norm(xj - xi))
        T = build_T(R)
        k_loc = elastic_stiffness_local(E, nu, A, Iy, Iz, J, L)
        k_glob = T.T @ k_loc @ T
        dofs = [dof_index(i, 0), dof_index(i, 1), dof_index(i, 2), dof_index(i, 3), dof_index(i, 4), dof_index(i, 5), dof_index(j, 0), dof_index(j, 1), dof_index(j, 2), dof_index(j, 3), dof_index(j, 4), dof_index(j, 5)]
        for a in range(12):
            ia = dofs[a]
            Ka = K[ia]
            for b in range(12):
                ib = dofs[b]
                Ka[ib] += k_glob[a, b]
        elem_data.append((i, j, L, T, dofs, E, A))
    K_ff = K[np.ix_(free_dofs, free_dofs)]
    P_f = P[free_dofs]
    try:
        u_f = np.linalg.solve(K_ff, P_f)
    except np.linalg.LinAlgError as e:
        raise ValueError(f'Linear system solve failed (check BCs and rigidity): {e}')
    u = np.zeros(ndof, dtype=float)
    u[free_dofs] = u_f
    K_g = np.zeros((ndof, ndof), dtype=float)
    total_norm_kge = 0.0
    for (i, j, L, T, dofs, E, A) in elem_data:
        d_e = u[dofs]
        d_loc = T @ d_e
        d_loc = T.T @ d_e
        u_i_local = d_loc[0]
        u_j_local = d_loc[6]
        N_tension = E * A / L * (u_j_local - u_i_local)
        N_comp = -float(N_tension)
        kG_loc = geometric_stiffness_local(N_comp, L)
        if np.any(kG_loc):
            kG_glob = T.T @ kG_loc @ T
            total_norm_kge += float(np.linalg.norm(kG_glob, ord='fro'))
            for a in range(12):
                ia = dofs[a]
                Kgrow = K_g[ia]
                for b in range(12):
                    ib = dofs[b]
                    Kgrow[ib] += kG_glob[a, b]
    K_g_ff = K_g[np.ix_(free_dofs, free_dofs)]
    if not np.isfinite(K_g_ff).all():
        raise ValueError('Geometric stiffness contains non-finite values')
    if total_norm_kge == 0.0 or np.linalg.norm(K_g_ff, ord='fro') <= 1e-20:
        raise ValueError('Geometric stiffness is zero; no buckling prediction possible under the given reference load.')
    try:
        (evals, evecs) = scipy.linalg.eig(K_ff, -K_g_ff)
    except Exception as e:
        raise ValueError(f'Generalized eigenproblem solve failed: {e}')
    evals = np.array(evals).ravel()
    evecs = np.array(evecs)
    imag_tol = 1e-08
    real_eigs = []
    real_vecs = []
    for k in range(evals.size):
        lam = evals[k]
        if abs(lam.imag) <= imag_tol * max(1.0, abs(lam.real)):
            val = float(lam.real)
            if val > 0.0:
                real_eigs.append(val)
                real_vecs.append(np.array(evecs[:, k].real, dtype=float))
    if len(real_eigs) == 0:
        raise ValueError('No positive real eigenvalue found for the buckling problem')
    real_eigs = np.array(real_eigs, dtype=float)
    idx_min = int(np.argmin(real_eigs))
    lambda_min = float(real_eigs[idx_min])
    phi_f = real_vecs[idx_min]
    phi = np.zeros(ndof, dtype=float)
    phi[free_dofs] = phi_f
    return (lambda_min, phi)