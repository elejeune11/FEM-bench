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
    ndof = 6 * n_nodes
    u_global = np.asarray(u_global, dtype=float).reshape(-1)
    if u_global.size != ndof:
        raise ValueError('u_global must have length 6 * n_nodes.')
    K_global = np.zeros((ndof, ndof), dtype=float)
    tol_zero = 1e-14
    tol_parallel = 1e-08
    tol_unit = 1e-06

    def assemble_global(K, edofs, k_e):
        np.add.at(K, (np.repeat(edofs, edofs.size), np.tile(edofs, edofs.size)), k_e.ravel())
    for el in elements:
        i = int(el['node_i'])
        j = int(el['node_j'])
        if not (0 <= i < n_nodes and 0 <= j < n_nodes):
            raise IndexError('Element node indices out of bounds.')
        E = float(el['E'])
        nu = float(el['nu'])
        A = float(el['A'])
        Iy = float(el['I_y'])
        Iz = float(el['I_z'])
        J = float(el['J'])
        xi = node_coords[i]
        xj = node_coords[j]
        dx = xj - xi
        L = float(np.linalg.norm(dx))
        if not L > tol_zero:
            raise ValueError('Zero-length element encountered.')
        ex = dx / L
        z_ref = el.get('local_z', None)
        if z_ref is None:
            z_try = np.array([0.0, 0.0, 1.0], dtype=float)
            if abs(float(np.dot(ex, z_try))) > 1.0 - tol_parallel:
                z_try = np.array([0.0, 1.0, 0.0], dtype=float)
            z_ref_vec = z_try
        else:
            z_ref_vec = np.asarray(z_ref, dtype=float).reshape(-1)
            if z_ref_vec.size != 3:
                raise ValueError('local_z must be a 3-vector.')
            nz = float(np.linalg.norm(z_ref_vec))
            if not abs(nz - 1.0) <= tol_unit:
                raise ValueError('local_z must be unit length.')
            if abs(float(np.dot(z_ref_vec, ex))) > 1.0 - tol_parallel:
                raise ValueError('local_z must not be parallel to the element axis.')
        y_temp = np.cross(z_ref_vec, ex)
        ny = float(np.linalg.norm(y_temp))
        if ny <= tol_zero:
            raise ValueError('Invalid local_z provided; cross product nearly zero.')
        ey = y_temp / ny
        ez = np.cross(ex, ey)
        R = np.vstack((ex, ey, ez))
        T = np.zeros((12, 12), dtype=float)
        for b in range(4):
            T[3 * b:3 * (b + 1), 3 * b:3 * (b + 1)] = R
        edofs = np.array([6 * i + k for k in range(6)] + [6 * j + k for k in range(6)], dtype=int)
        u_e_global = u_global[edofs]
        u_local = T.T @ u_e_global
        G = E / (2.0 * (1.0 + nu))
        L2 = L * L
        L3 = L2 * L
        Ke = np.zeros((12, 12), dtype=float)
        k_ax = E * A / L
        Ke[0, 0] += k_ax
        Ke[0, 6] -= k_ax
        Ke[6, 0] -= k_ax
        Ke[6, 6] += k_ax
        k_tor = G * J / L
        Ke[3, 3] += k_tor
        Ke[3, 9] -= k_tor
        Ke[9, 3] -= k_tor
        Ke[9, 9] += k_tor
        a_z = E * Iz
        kbz = np.array([[12.0 * a_z / L3, 6.0 * a_z / L2, -12.0 * a_z / L3, 6.0 * a_z / L2], [6.0 * a_z / L2, 4.0 * a_z / L, -6.0 * a_z / L2, 2.0 * a_z / L], [-12.0 * a_z / L3, -6.0 * a_z / L2, 12.0 * a_z / L3, -6.0 * a_z / L2], [6.0 * a_z / L2, 2.0 * a_z / L, -6.0 * a_z / L2, 4.0 * a_z / L]], dtype=float)
        idx_z = [1, 5, 7, 11]
        for a in range(4):
            for b in range(4):
                Ke[idx_z[a], idx_z[b]] += kbz[a, b]
        a_y = E * Iy
        kby = np.array([[12.0 * a_y / L3, -6.0 * a_y / L2, -12.0 * a_y / L3, -6.0 * a_y / L2], [-6.0 * a_y / L2, 4.0 * a_y / L, 6.0 * a_y / L2, 2.0 * a_y / L], [-12.0 * a_y / L3, 6.0 * a_y / L2, 12.0 * a_y / L3, 6.0 * a_y / L2], [-6.0 * a_y / L2, 2.0 * a_y / L, 6.0 * a_y / L2, 4.0 * a_y / L]], dtype=float)
        idx_y = [2, 4, 8, 10]
        for a in range(4):
            for b in range(4):
                Ke[idx_y[a], idx_y[b]] += kby[a, b]
        f_local = Ke @ u_local
        N = 0.5 * (f_local[6] - f_local[0])
        Kg_local = np.zeros((12, 12), dtype=float)
        if abs(N) > 0.0:
            c = N / (30.0 * L)
            L_ = L
            L2_ = L * L
            kg_block = np.array([[36.0, 3.0 * L_, -36.0, 3.0 * L_], [3.0 * L_, 4.0 * L2_, -3.0 * L_, -1.0 * L2_], [-36.0, -3.0 * L_, 36.0, -3.0 * L_], [3.0 * L_, -1.0 * L2_, -3.0 * L_, 4.0 * L2_]], dtype=float) * c
            for idxs in (idx_z, idx_y):
                for a in range(4):
                    ia = idxs[a]
                    for b in range(4):
                        ib = idxs[b]
                        Kg_local[ia, ib] += kg_block[a, b]
        Kg_global_e = T @ Kg_local @ T.T
        assemble_global(K_global, edofs, Kg_global_e)
    return K_global