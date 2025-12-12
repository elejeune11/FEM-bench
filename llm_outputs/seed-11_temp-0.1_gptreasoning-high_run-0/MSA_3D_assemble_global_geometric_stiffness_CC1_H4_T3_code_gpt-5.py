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
    if node_coords.ndim != 2 or node_coords.shape[1] != 3:
        raise ValueError('node_coords must be an array of shape (n_nodes, 3).')
    n_nodes = node_coords.shape[0]
    n_dof = 6 * n_nodes
    u_global = np.asarray(u_global, dtype=float).reshape(-1)
    if u_global.size != n_dof:
        raise ValueError('u_global length must be 6 * n_nodes.')
    if elements is None:
        return np.zeros((n_dof, n_dof), dtype=float)
    K_global = np.zeros((n_dof, n_dof), dtype=float)
    tol = 1e-12
    unit_z = np.array([0.0, 0.0, 1.0], dtype=float)
    unit_y = np.array([0.0, 1.0, 0.0], dtype=float)
    for e in elements:
        try:
            i = int(e['node_i'])
            j = int(e['node_j'])
            E = float(e['E'])
            nu = float(e['nu'])
            A = float(e['A'])
            Iy = float(e['I_y'])
            Iz = float(e['I_z'])
            J = float(e['J'])
        except KeyError as ke:
            raise KeyError(f'Missing required element key: {ke}')
        if not (0 <= i < n_nodes and 0 <= j < n_nodes):
            raise IndexError('Element node indices out of range.')
        xi = node_coords[i]
        xj = node_coords[j]
        dx = xj - xi
        L = float(np.linalg.norm(dx))
        if not np.isfinite(L) or L <= tol:
            raise ValueError('Zero or invalid length element detected.')
        x_dir = dx / L
        z_ref = e.get('local_z', None)
        if z_ref is None:
            if abs(float(np.dot(x_dir, unit_z))) >= 1.0 - 1e-08:
                z_ref = unit_y.copy()
            else:
                z_ref = unit_z.copy()
        else:
            z_ref = np.asarray(z_ref, dtype=float).reshape(3)
            norm_z = float(np.linalg.norm(z_ref))
            if norm_z <= tol:
                raise ValueError('Provided local_z has zero length.')
            if abs(norm_z - 1.0) > 1e-06:
                raise ValueError('Provided local_z must be unit length.')
            if abs(float(np.dot(z_ref, x_dir))) >= 1.0 - 1e-08:
                raise ValueError('Provided local_z must not be parallel to the element axis.')
        y_temp = np.cross(z_ref, x_dir)
        ny = float(np.linalg.norm(y_temp))
        if ny <= tol:
            raise ValueError('Invalid local_z: parallel or nearly parallel to element axis.')
        y_dir = y_temp / ny
        z_dir = np.cross(x_dir, y_dir)
        nz = float(np.linalg.norm(z_dir))
        if nz <= tol:
            raise ValueError('Failed to construct local axes (degenerate).')
        z_dir = z_dir / nz
        R = np.vstack((x_dir, y_dir, z_dir))
        T = np.zeros((12, 12), dtype=float)
        T[0:3, 0:3] = R
        T[3:6, 3:6] = R
        T[6:9, 6:9] = R
        T[9:12, 9:12] = R
        G = E / (2.0 * (1.0 + nu))
        L2 = L * L
        L3 = L2 * L
        ke = np.zeros((12, 12), dtype=float)
        EA_L = E * A / L
        ke[0, 0] = EA_L
        ke[0, 6] = -EA_L
        ke[6, 0] = -EA_L
        ke[6, 6] = EA_L
        GJ_L = G * J / L
        ke[3, 3] = GJ_L
        ke[3, 9] = -GJ_L
        ke[9, 3] = -GJ_L
        ke[9, 9] = GJ_L
        c1 = 12.0 * E * Iz / L3
        c2 = 6.0 * E * Iz / L2
        c3 = 4.0 * E * Iz / L
        c4 = 2.0 * E * Iz / L
        ke[1, 1] += c1
        ke[1, 5] += c2
        ke[1, 7] += -c1
        ke[1, 11] += c2
        ke[5, 1] += c2
        ke[5, 5] += c3
        ke[5, 7] += -c2
        ke[5, 11] += c4
        ke[7, 1] += -c1
        ke[7, 5] += -c2
        ke[7, 7] += c1
        ke[7, 11] += -c2
        ke[11, 1] += c2
        ke[11, 5] += c4
        ke[11, 7] += -c2
        ke[11, 11] += c3
        d1 = 12.0 * E * Iy / L3
        d2 = 6.0 * E * Iy / L2
        d3 = 4.0 * E * Iy / L
        d4 = 2.0 * E * Iy / L
        ke[2, 2] += d1
        ke[2, 4] += -d2
        ke[2, 8] += -d1
        ke[2, 10] += -d2
        ke[4, 2] += -d2
        ke[4, 4] += d3
        ke[4, 8] += d2
        ke[4, 10] += d4
        ke[8, 2] += -d1
        ke[8, 4] += d2
        ke[8, 8] += d1
        ke[8, 10] += d2
        ke[10, 2] += -d2
        ke[10, 4] += d4
        ke[10, 8] += d2
        ke[10, 10] += d3
        i0 = 6 * i
        j0 = 6 * j
        d_g = np.concatenate((u_global[i0:i0 + 6], u_global[j0:j0 + 6]))
        d_l = T @ d_g
        f_l = ke @ d_l
        N = 0.5 * (f_l[0] - f_l[6])
        k_g_local = np.zeros((12, 12), dtype=float)
        if abs(L) > tol and np.isfinite(N):
            a = N / (30.0 * L)
            Lterm = L
            L2term = L * L
            K4 = a * np.array([[36.0, 3.0 * Lterm, -36.0, 3.0 * Lterm], [3.0 * Lterm, 4.0 * L2term, -3.0 * Lterm, -1.0 * L2term], [-36.0, -3.0 * Lterm, 36.0, -3.0 * Lterm], [3.0 * Lterm, -1.0 * L2term, -3.0 * Lterm, 4.0 * L2term]], dtype=float)
            idx_vrz = [1, 5, 7, 11]
            for r in range(4):
                for c in range(4):
                    k_g_local[idx_vrz[r], idx_vrz[c]] += K4[r, c]
            idx_wry = [2, 4, 8, 10]
            for r in range(4):
                for c in range(4):
                    k_g_local[idx_wry[r], idx_wry[c]] += K4[r, c]
        k_g_global_e = T.T @ k_g_local @ T
        edofs = list(range(i0, i0 + 6)) + list(range(j0, j0 + 6))
        K_global[np.ix_(edofs, edofs)] += k_g_global_e
    return K_global