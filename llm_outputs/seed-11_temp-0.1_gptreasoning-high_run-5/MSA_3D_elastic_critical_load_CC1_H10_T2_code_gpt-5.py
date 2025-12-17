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
    import numpy as _np
    import scipy as _scipy
    coords = _np.asarray(node_coords, dtype=float)
    if coords.ndim != 2 or coords.shape[1] != 3:
        raise ValueError('node_coords must be an (N,3) array of floats')
    n_nodes = coords.shape[0]
    ndof = 6 * n_nodes

    def _node_dofs(n):
        b = 6 * n
        return _np.arange(b, b + 6, dtype=int)
    P = _np.zeros(ndof, dtype=float)
    if nodal_loads is not None:
        for ni, load in nodal_loads.items():
            if ni < 0 or ni >= n_nodes:
                raise ValueError('nodal_loads contains invalid node index')
            load = _np.asarray(load, dtype=float)
            if load.size != 6:
                raise ValueError('Each nodal load must have 6 components [Fx, Fy, Fz, Mx, My, Mz]')
            P[_node_dofs(ni)] += load
    fixed = _np.zeros(ndof, dtype=bool)
    if boundary_conditions is not None:
        for ni, bc in boundary_conditions.items():
            if ni < 0 or ni >= n_nodes:
                raise ValueError('boundary_conditions contains invalid node index')
            bc = _np.asarray(bc, dtype=int)
            if bc.size != 6:
                raise ValueError('Each boundary condition vector must have 6 entries of 0/1')
            dofs = _node_dofs(ni)
            fixed[dofs] = fixed[dofs] | bc.astype(bool)

    def _element_rotation_and_T(xi, xj, local_z_hint=None):
        L_vec = xj - xi
        L = _np.linalg.norm(L_vec)
        if not _np.isfinite(L) or L <= 0.0:
            raise ValueError('Element length must be positive')
        ex = L_vec / L
        tol_ax = 1e-12
        if local_z_hint is None:
            z_guess = _np.array([0.0, 0.0, 1.0], dtype=float)
            if abs(_np.dot(ex, z_guess)) > 1.0 - 1e-08:
                z_guess = _np.array([0.0, 1.0, 0.0], dtype=float)
        else:
            z_guess = _np.asarray(local_z_hint, dtype=float)
            if z_guess.size != 3:
                raise ValueError('local_z must be a 3-component vector')
            nrm = _np.linalg.norm(z_guess)
            if nrm <= tol_ax:
                raise ValueError('local_z has near-zero norm')
            z_guess = z_guess / nrm
            if abs(_np.dot(ex, z_guess)) > 1.0 - 1e-08:
                z_guess = _np.array([0.0, 0.0, 1.0], dtype=float)
                if abs(_np.dot(ex, z_guess)) > 1.0 - 1e-08:
                    z_guess = _np.array([0.0, 1.0, 0.0], dtype=float)
        ey_temp = _np.cross(z_guess, ex)
        n_ey = _np.linalg.norm(ey_temp)
        if n_ey <= tol_ax:
            alt = _np.array([0.0, 1.0, 0.0], dtype=float) if abs(ex[1]) < 0.9 else _np.array([1.0, 0.0, 0.0], dtype=float)
            ey_temp = _np.cross(alt, ex)
            n_ey = _np.linalg.norm(ey_temp)
            if n_ey <= tol_ax:
                raise ValueError('Failed to construct orthonormal local basis')
        ey = ey_temp / n_ey
        ez = _np.cross(ex, ey)
        ez /= _np.linalg.norm(ez)
        R = _np.vstack((ex, ey, ez))
        T = _np.zeros((12, 12), dtype=float)
        T[0:3, 0:3] = R
        T[3:6, 3:6] = R
        T[6:9, 6:9] = R
        T[9:12, 9:12] = R
        return (R, T, L)

    def _local_elastic_stiffness(E, nu, A, I_y, I_z, J, L):
        K = _np.zeros((12, 12), dtype=float)
        k_ax = E * A / L
        K[0, 0] += k_ax
        K[0, 6] -= k_ax
        K[6, 0] -= k_ax
        K[6, 6] += k_ax
        G = E / (2.0 * (1.0 + nu))
        k_t = G * J / L
        K[3, 3] += k_t
        K[3, 9] -= k_t
        K[9, 3] -= k_t
        K[9, 9] += k_t

        def add_bending_block(EI, v1, rz1, v2, rz2):
            Lc = L
            k11 = 12.0 * EI / Lc ** 3
            k12 = 6.0 * EI / Lc ** 2
            k22 = 4.0 * EI / Lc
            k24 = 2.0 * EI / Lc
            K[v1, v1] += k11
            K[v1, rz1] += k12
            K[v1, v2] += -k11
            K[v1, rz2] += k12
            K[rz1, v1] += k12
            K[rz1, rz1] += k22
            K[rz1, v2] += -k12
            K[rz1, rz2] += k24
            K[v2, v1] += -k11
            K[v2, rz1] += -k12
            K[v2, v2] += k11
            K[v2, rz2] += -k12
            K[rz2, v1] += k12
            K[rz2, rz1] += k24
            K[rz2, v2] += -k12
            K[rz2, rz2] += k22
        add_bending_block(E * I_z, 1, 5, 7, 11)
        add_bending_block(E * I_y, 2, 4, 8, 10)
        return K
    K = _np.zeros((ndof, ndof), dtype=float)
    elem_cache = []
    for e in elements:
        for key in ('node_i', 'node_j', 'E', 'nu', 'A', 'I_y', 'I_z', 'J'):
            if key not in e:
                raise ValueError(f"Element is missing required key '{key}'")
        ni = int(e['node_i'])
        nj = int(e['node_j'])
        if ni < 0 or nj < 0 or ni >= n_nodes or (nj >= n_nodes) or (ni == nj):
            raise ValueError('Invalid element node indices')
        xi = coords[ni]
        xj = coords[nj]
        local_z_hint = e.get('local_z', None)
        _, T_e, L_e = _element_rotation_and_T(xi, xj, local_z_hint)
        K_loc = _local_elastic_stiffness(float(e['E']), float(e['nu']), float(e['A']), float(e['I_y']), float(e['I_z']), float(e['J']), L_e)
        K_gl = T_e.T @ K_loc @ T_e
        idx_i = _node_dofs(ni)
        idx_j = _node_dofs(nj)
        idx_e = _np.concatenate((idx_i, idx_j))
        K[_np.ix_(idx_e, idx_e)] += K_gl
        elem_cache.append({'indices': idx_e, 'T': T_e, 'K_local': K_loc, 'L': L_e, 'A': float(e['A']), 'J': float(e['J'])})
    free = _np.where(~fixed)[0]
    if free.size == 0:
        raise ValueError('All DOFs are fixed; no free DOFs to analyze')
    K_ff = K[_np.ix_(free, free)]
    P_f = P[free]
    try:
        u_f = _scipy.linalg.solve(K_ff, P_f, assume_a='sym', check_finite=False)
    except Exception as exc:
        raise ValueError(f'Static solve failed; check boundary conditions and stiffness. Details: {exc}') from exc
    u = _np.zeros(ndof, dtype=float)
    u[free] = u_f
    K_g = _np.zeros((ndof, ndof), dtype=float)
    for ec in elem_cache:
        idx_e = ec['indices']
        T_e = ec['T']
        d_e_global = u[idx_e]
        d_e_local = T_e @ d_e_global
        f_e_local = ec['K_local'] @ d_e_local
        Fx2 = float(f_e_local[6])
        Mx2 = float(f_e_local[9])
        My1 = float(f_e_local[4])
        Mz1 = float(f_e_local[5])
        My2 = float(f_e_local[10])
        Mz2 = float(f_e_local[11])
        k_g_loc = local_geometric_stiffness_matrix_3D_beam(L=ec['L'], A=ec['A'], I_rho=ec['J'], Fx2=Fx2, Mx2=Mx2, My1=My1, Mz1=Mz1, My2=My2, Mz2=Mz2)
        k_g_gl = T_e.T @ k_g_loc @ T_e
        K_g[_np.ix_(idx_e, idx_e)] += k_g_gl
    K_g_ff = K_g[_np.ix_(free, free)]
    if _np.linalg.norm(K_g_ff, ord='fro') <= 1e-14 * max(1.0, _np.linalg.norm(K_ff, ord='fro')):
        raise ValueError('Geometric stiffness is nearly zero under the reference load; cannot perform buckling analysis')
    try:
        C = -_scipy.linalg.solve(K_ff, K_g_ff, assume_a='sym', check_finite=False)
        eigvals, eigvecs = _scipy.linalg.eig(C, check_finite=False)
    except Exception as exc:
        raise ValueError(f'Eigenvalue solve failed; details: {exc}') from exc
    real_parts = eigvals.real
    imag_parts = eigvals.imag
    scale = _np.maximum(1.0, _np.abs(real_parts))
    ok_real = _np.abs(imag_parts) <= 1e-07 * scale
    candidates_mask = ok_real & (real_parts > 1e-10)
    if not _np.any(candidates_mask):
        candidates_mask = ok_real & (real_parts > 1e-12)
    if not _np.any(candidates_mask):
        raise ValueError('No positive real eigenvalue found for buckling')
    pos_vals = real_parts[candidates_mask]
    idx_candidates = _np.nonzero(candidates_mask)[0]
    min_idx_local = _np.argmin(pos_vals)
    eig_index = idx_candidates[min_idx_local]
    lambda_min = float(real_parts[eig_index])
    phi_f = eigvecs[:, eig_index]
    if _np.abs(phi_f.imag).max() > 1e-06 * max(1.0, _np.abs(phi_f.real).max()):
        raise ValueError('Selected buckling mode is significantly complex')
    phi_f = phi_f.real
    phi = _np.zeros(ndof, dtype=float)
    phi[free] = phi_f
    return (lambda_min, phi)