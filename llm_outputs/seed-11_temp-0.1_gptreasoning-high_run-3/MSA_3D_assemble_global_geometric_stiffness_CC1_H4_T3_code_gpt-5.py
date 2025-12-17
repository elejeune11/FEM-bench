def MSA_3D_assemble_global_geometric_stiffness_CC1_H4_T3(node_coords: np.ndarray, elements: Sequence[dict], u_global: np.ndarray) -> np.ndarray:
    """
    Assemble the global geometric (initial-stress) stiffness matrix K_g for a 3D frame
    under a given global displacement state.
    Each 2-node Euler–Bernoulli beam contributes a 12×12 local geometric stiffness
    matrix k_g^local that depends on the element length and the internal end
    force/moment resultants induced by the current displacement state. The local
    matrix is then mapped to global coordinates with a 12×12 direction-cosine
    transformation Γ and scattered into the global K_g.
    Parameters
    ----------
    node_coords : (n_nodes, 3) ndarray of float
        Global Cartesian coordinates [x, y, z] of each node (0-based indexing).
    elements : sequence of dict
        Per-element dictionaries. Required keys per element:
            'node_i', 'node_j' : int
                Indices of the start and end nodes.
            'E' : float
                Young's modulus (Pa).
            'nu' : float
                Poisson's ratio (unitless).
            'A' : float
                Cross-sectional area (m²).
            'I_y', 'I_z' : float
                Second moments of area about the local y- and z-axes (m⁴).
            'J' : float
                Torsional constant (m⁴).
            'local_z' : array-like of shape (3,), optional
                Unit vector in global coordinates defining the local z-axis orientation.
                Must not be parallel to the beam axis. If None, a default is chosen (see Notes).
    u_global : (6*n_nodes,) ndarray of float
        Global displacement vector with 6 DOF per node in the order
        [u_x, u_y, u_z, θ_x, θ_y, θ_z] for node 0, then node 1, etc.
    Returns
    -------
    K : (6*n_nodes, 6*n_nodes) ndarray of float
        Assembled global geometric stiffness matrix. For conservative loading and
        the standard formulation, K_g is symmetric.
    Notes
    -----
      unless the beam axis is aligned with global z, in which case use the global y-axis.
      The 'local_z' must be unit length and not parallel to the beam axis.
      induced by the supplied displacement state (not external loads). Their local DOF
      ordering is the same as for local displacements:
      [u1, v1, w1, θx1, θy1, θz1, u2, v2, w2, θx2, θy2, θz2] ↔
      [Fx_i, Fy_i, Fz_i, Mx_i, My_i, Mz_i, Fx_j, Fy_j, Fz_j, Mx_j, My_j, Mz_j].
      should be treated as an error by the transformation routine.
    Effects captured in the geometric stiffness matrix:
        + **Tension (+Fx2)** increases lateral/torsional stiffness.
        + Compression (-Fx2) decreases it and may trigger buckling when K_e + K_g becomes singular.
    """
    node_coords = np.asarray(node_coords, dtype=float)
    u_global = np.asarray(u_global, dtype=float)
    if node_coords.ndim != 2 or node_coords.shape[1] != 3:
        raise ValueError('node_coords must be of shape (n_nodes, 3).')
    n_nodes = node_coords.shape[0]
    if u_global.shape != (6 * n_nodes,):
        raise ValueError('u_global must be a 1D array of length 6*n_nodes.')
    K = np.zeros((6 * n_nodes, 6 * n_nodes), dtype=float)
    tol_len = 1e-12
    tol_unit = 1e-06
    tol_parallel = 1.0 - 1e-08
    for elem in elements:
        ni = int(elem['node_i'])
        nj = int(elem['node_j'])
        if not (0 <= ni < n_nodes and 0 <= nj < n_nodes):
            raise ValueError('Element node indices out of range.')
        xi = node_coords[ni]
        xj = node_coords[nj]
        dx = xj - xi
        L = float(np.linalg.norm(dx))
        if not np.isfinite(L) or L <= tol_len:
            raise ValueError('Zero-length or invalid element length encountered.')
        ex = dx / L
        zref = elem.get('local_z', None)
        if zref is None:
            zref = np.array([0.0, 0.0, 1.0], dtype=float)
            if abs(np.dot(zref, ex)) >= tol_parallel:
                zref = np.array([0.0, 1.0, 0.0], dtype=float)
        else:
            zref = np.asarray(zref, dtype=float)
            if zref.shape != (3,):
                raise ValueError('local_z must be array-like with 3 components.')
            nz = float(np.linalg.norm(zref))
            if not np.isfinite(nz) or nz <= tol_len:
                raise ValueError('local_z must be a non-zero vector.')
            if abs(nz - 1.0) > tol_unit:
                raise ValueError('local_z must be unit length.')
            if abs(np.dot(zref, ex)) >= tol_parallel:
                raise ValueError('local_z must not be parallel to the element axis.')
        ey = np.cross(zref, ex)
        ney = float(np.linalg.norm(ey))
        if ney <= tol_len:
            raise ValueError('local_z produced degenerate local basis (parallel to axis).')
        ey /= ney
        ez = np.cross(ex, ey)
        A = np.column_stack((ex, ey, ez))
        R = A.T
        Gamma = np.zeros((12, 12), dtype=float)
        Gamma[0:3, 0:3] = R
        Gamma[3:6, 3:6] = R
        Gamma[6:9, 6:9] = R
        Gamma[9:12, 9:12] = R
        edofs = np.concatenate((np.arange(ni * 6, ni * 6 + 6), np.arange(nj * 6, nj * 6 + 6)))
        u_e_g = u_global[edofs]
        u_e_l = Gamma @ u_e_g
        E = float(elem['E'])
        A_cs = float(elem['A'])
        N = E * A_cs / L * (u_e_l[6] - u_e_l[0])
        c = N / (30.0 * L)
        L2 = L * L
        kg2d = c * np.array([[36.0, 3.0 * L, -36.0, 3.0 * L], [3.0 * L, 4.0 * L2, -3.0 * L, -1.0 * L2], [-36.0, -3.0 * L, 36.0, -3.0 * L], [3.0 * L, -1.0 * L2, -3.0 * L, 4.0 * L2]], dtype=float)
        k_g_local = np.zeros((12, 12), dtype=float)
        idx_v = [1, 5, 7, 11]
        k_g_local[np.ix_(idx_v, idx_v)] += kg2d
        idx_w = [2, 4, 8, 10]
        k_g_local[np.ix_(idx_w, idx_w)] += kg2d
        k_g_global = Gamma.T @ k_g_local @ Gamma
        K[np.ix_(edofs, edofs)] += k_g_global
    return K