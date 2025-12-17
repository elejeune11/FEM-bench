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
    if not isinstance(node_coords, np.ndarray) or node_coords.ndim != 2 or node_coords.shape[1] != 3:
        raise ValueError('node_coords must be an (n_nodes, 3) ndarray.')
    n_nodes = node_coords.shape[0]
    if not isinstance(u_global, np.ndarray) or u_global.ndim != 1 or u_global.size != 6 * n_nodes:
        raise ValueError('u_global must be a 1D ndarray of length 6*n_nodes.')
    K_global = np.zeros((6 * n_nodes, 6 * n_nodes), dtype=float)

    def _rotation_matrix(i_node: int, j_node: int, local_z_vec_optional) -> np.ndarray:
        pi = node_coords[i_node]
        pj = node_coords[j_node]
        dx = pj - pi
        L = np.linalg.norm(dx)
        if not np.isfinite(L) or L <= 0.0:
            raise ValueError('Zero-length element encountered.')
        ex = dx / L
        if local_z_vec_optional is None:
            zref = np.array([0.0, 0.0, 1.0], dtype=float)
            if abs(np.dot(ex, zref)) > 1.0 - 1e-08:
                zref = np.array([0.0, 1.0, 0.0], dtype=float)
        else:
            zref = np.asarray(local_z_vec_optional, dtype=float).reshape(-1)
            if zref.size != 3:
                raise ValueError('local_z must be a length-3 vector.')
            nz = np.linalg.norm(zref)
            if not np.isfinite(nz) or abs(nz - 1.0) > 1e-08:
                raise ValueError('Provided local_z must be unit length.')
            if abs(np.dot(ex, zref)) > 1.0 - 1e-08:
                raise ValueError('Provided local_z cannot be parallel to the element axis.')
        ey = np.cross(zref, ex)
        ney = np.linalg.norm(ey)
        if not np.isfinite(ney) or ney <= 1e-12:
            raise ValueError('Invalid local_z; cannot construct orthonormal triad.')
        ey = ey / ney
        ez = np.cross(ex, ey)
        nez = np.linalg.norm(ez)
        if not np.isfinite(nez) or nez <= 1e-12:
            raise ValueError('Failed to construct right-handed local axes.')
        ez = ez / nez
        R = np.column_stack((ex, ey, ez))
        return (R, L)

    def _T_g2l(R: np.ndarray) -> np.ndarray:
        T = np.zeros((12, 12), dtype=float)
        Rt = R.T
        T[0:3, 0:3] = Rt
        T[3:6, 3:6] = Rt
        T[6:9, 6:9] = Rt
        T[9:12, 9:12] = Rt
        return T

    def _linear_local_stiffness(E, G, A, Iy, Iz, J, L) -> np.ndarray:
        K = np.zeros((12, 12), dtype=float)
        k_ax = E * A / L
        K[0, 0] += k_ax
        K[0, 6] += -k_ax
        K[6, 0] += -k_ax
        K[6, 6] += k_ax
        k_t = G * J / L
        K[3, 3] += k_t
        K[3, 9] += -k_t
        K[9, 3] += -k_t
        K[9, 9] += k_t
        EI_z = E * Iz
        L2 = L * L
        L3 = L2 * L
        kbz = EI_z / L3 * np.array([[12.0, 6.0 * L, -12.0, 6.0 * L], [6.0 * L, 4.0 * L2, -6.0 * L, 2.0 * L2], [-12.0, -6.0 * L, 12.0, -6.0 * L], [6.0 * L, 2.0 * L2, -6.0 * L, 4.0 * L2]], dtype=float)
        idx_vz = [1, 5, 7, 11]
        for a in range(4):
            for b in range(4):
                K[idx_vz[a], idx_vz[b]] += kbz[a, b]
        EI_y = E * Iy
        kby = EI_y / L3 * np.array([[12.0, 6.0 * L, -12.0, 6.0 * L], [6.0 * L, 4.0 * L2, -6.0 * L, 2.0 * L2], [-12.0, -6.0 * L, 12.0, -6.0 * L], [6.0 * L, 2.0 * L2, -6.0 * L, 4.0 * L2]], dtype=float)
        idx_wy = [2, 4, 8, 10]
        for a in range(4):
            for b in range(4):
                K[idx_wy[a], idx_wy[b]] += kby[a, b]
        return K

    def _geom_local_stiffness_axial(N, L) -> np.ndarray:
        Kg = np.zeros((12, 12), dtype=float)
        if L <= 0.0:
            return Kg
        c = N / (30.0 * L)
        L2 = L * L
        ksub = c * np.array([[36.0, 3.0 * L, -36.0, 3.0 * L], [3.0 * L, 4.0 * L2, -3.0 * L, -1.0 * L2], [-36.0, -3.0 * L, 36.0, -3.0 * L], [3.0 * L, -1.0 * L2, -3.0 * L, 4.0 * L2]], dtype=float)
        idx_vz = [1, 5, 7, 11]
        for a in range(4):
            for b in range(4):
                Kg[idx_vz[a], idx_vz[b]] += ksub[a, b]
        idx_wy = [2, 4, 8, 10]
        for a in range(4):
            for b in range(4):
                Kg[idx_wy[a], idx_wy[b]] += ksub[a, b]
        return Kg
    for e in elements:
        i = int(e['node_i'])
        j = int(e['node_j'])
        if i < 0 or i >= n_nodes or j < 0 or (j >= n_nodes):
            raise IndexError('Element node index out of bounds.')
        local_z_vec = e.get('local_z', None)
        R, L = _rotation_matrix(i, j, local_z_vec)
        T_g2l = _T_g2l(R)
        E = float(e['E'])
        nu = float(e['nu'])
        A = float(e['A'])
        Iy = float(e['I_y'])
        Iz = float(e['I_z'])
        J = float(e['J'])
        if not np.isfinite(L) or L <= 0.0:
            raise ValueError('Invalid element length.')
        G = E / (2.0 * (1.0 + nu))
        dofs_i = np.arange(6 * i, 6 * i + 6, dtype=int)
        dofs_j = np.arange(6 * j, 6 * j + 6, dtype=int)
        edofs = np.concatenate((dofs_i, dofs_j))
        u_e_g = u_global[edofs]
        u_e_l = T_g2l @ u_e_g
        K_e_loc = _linear_local_stiffness(E, G, A, Iy, Iz, J, L)
        q_int_loc = K_e_loc @ u_e_l
        N = 0.5 * (q_int_loc[6] - q_int_loc[0])
        K_g_loc = _geom_local_stiffness_axial(N, L)
        K_g_glob_e = T_g2l.T @ K_g_loc @ T_g2l
        for a in range(12):
            ra = edofs[a]
            Ka_row = K_global[ra]
            for b in range(12):
                Ka_row[edofs[b]] += K_g_glob_e[a, b]
    return K_global