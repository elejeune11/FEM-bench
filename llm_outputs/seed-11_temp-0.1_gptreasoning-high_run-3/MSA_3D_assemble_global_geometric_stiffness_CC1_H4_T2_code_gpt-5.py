def MSA_3D_assemble_global_geometric_stiffness_CC1_H4_T2(node_coords: np.ndarray, elements: Sequence[dict], u_global: np.ndarray) -> np.ndarray:
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
    External Dependencies
    ---------------------
    local_geometric_stiffness_matrix_3D_beam(L, A, I_rho, Fx2, Mx2, My1, Mz1, My2, Mz2) -> (12,12) ndarray
        Must return the local geometric stiffness using the element length L, section properties, and local end force resultants as shown.
    """
    n_nodes = int(node_coords.shape[0])
    if node_coords.shape[1] != 3:
        raise ValueError('node_coords must have shape (n_nodes, 3).')
    ndof_total = 6 * n_nodes
    if u_global.shape[0] != ndof_total:
        raise ValueError('u_global length must be 6 * n_nodes.')
    K = np.zeros((ndof_total, ndof_total), dtype=float)
    tol_len = 1e-12
    tol_unit = 1e-08
    tol_parallel = 1e-08
    for el in elements:
        i = int(el['node_i'])
        j = int(el['node_j'])
        if not (0 <= i < n_nodes and 0 <= j < n_nodes):
            raise ValueError('Element node indices out of range.')
        xi = np.asarray(node_coords[i], dtype=float)
        xj = np.asarray(node_coords[j], dtype=float)
        d = xj - xi
        L = float(np.linalg.norm(d))
        if not np.isfinite(L) or L <= tol_len:
            raise ValueError('Zero or invalid element length.')
        ex = d / L
        z_ref = el.get('local_z', None)
        if z_ref is None:
            z_ref = np.array([0.0, 0.0, 1.0], dtype=float)
            if abs(float(np.dot(ex, z_ref))) >= 1.0 - tol_parallel:
                z_ref = np.array([0.0, 1.0, 0.0], dtype=float)
        else:
            z_ref = np.asarray(z_ref, dtype=float).reshape(-1)
            if z_ref.size != 3:
                raise ValueError('local_z must have 3 components.')
            nrm = float(np.linalg.norm(z_ref))
            if not np.isfinite(nrm) or abs(nrm - 1.0) > tol_unit:
                raise ValueError('local_z must be a unit vector.')
            if abs(float(np.dot(ex, z_ref))) >= 1.0 - tol_parallel:
                raise ValueError('local_z must not be parallel to the element axis.')
        y_tmp = np.cross(z_ref, ex)
        ny = float(np.linalg.norm(y_tmp))
        if ny <= tol_len:
            raise ValueError('Invalid local_z: parallel to element axis.')
        ey = y_tmp / ny
        ez = np.cross(ex, ey)
        R = np.column_stack((ex, ey, ez))
        RT = R.T
        T = np.zeros((12, 12), dtype=float)
        for k in range(4):
            T[k * 3:(k + 1) * 3, k * 3:(k + 1) * 3] = RT
        E = float(el['E'])
        nu = float(el['nu'])
        A = float(el['A'])
        Iy = float(el['I_y'])
        Iz = float(el['I_z'])
        J = float(el['J'])
        G = E / (2.0 * (1.0 + nu))
        k_loc = np.zeros((12, 12), dtype=float)
        k_ax = E * A / L
        k_loc[0, 0] += k_ax
        k_loc[0, 6] += -k_ax
        k_loc[6, 0] += -k_ax
        k_loc[6, 6] += k_ax
        k_tor = G * J / L
        k_loc[3, 3] += k_tor
        k_loc[3, 9] += -k_tor
        k_loc[9, 3] += -k_tor
        k_loc[9, 9] += k_tor
        EIz = E * Iz
        a = 12.0 * EIz / L ** 3
        b = 6.0 * EIz / L ** 2
        c = 4.0 * EIz / L
        dcoef = 2.0 * EIz / L
        k_loc[1, 1] += a
        k_loc[1, 5] += b
        k_loc[1, 7] += -a
        k_loc[1, 11] += b
        k_loc[5, 1] += b
        k_loc[5, 5] += c
        k_loc[5, 7] += -b
        k_loc[5, 11] += dcoef
        k_loc[7, 1] += -a
        k_loc[7, 5] += -b
        k_loc[7, 7] += a
        k_loc[7, 11] += -b
        k_loc[11, 1] += b
        k_loc[11, 5] += dcoef
        k_loc[11, 7] += -b
        k_loc[11, 11] += c
        EIy = E * Iy
        a = 12.0 * EIy / L ** 3
        b = 6.0 * EIy / L ** 2
        c = 4.0 * EIy / L
        dcoef = 2.0 * EIy / L
        k_loc[2, 2] += a
        k_loc[2, 4] += -b
        k_loc[2, 8] += -a
        k_loc[2, 10] += -b
        k_loc[4, 2] += -b
        k_loc[4, 4] += c
        k_loc[4, 8] += b
        k_loc[4, 10] += dcoef
        k_loc[8, 2] += -a
        k_loc[8, 4] += b
        k_loc[8, 8] += a
        k_loc[8, 10] += b
        k_loc[10, 2] += -b
        k_loc[10, 4] += dcoef
        k_loc[10, 8] += b
        k_loc[10, 10] += c
        dof_map = np.concatenate((np.arange(6 * i, 6 * i + 6), np.arange(6 * j, 6 * j + 6)))
        u_e_global = u_global[dof_map]
        u_e_local = T @ u_e_global
        f_local = k_loc @ u_e_local
        Fx2 = float(f_local[6])
        Mx2 = float(f_local[9])
        My1 = float(f_local[4])
        Mz1 = float(f_local[5])
        My2 = float(f_local[10])
        Mz2 = float(f_local[11])
        I_rho = Iy + Iz
        k_g_local = local_geometric_stiffness_matrix_3D_beam(L=L, A=A, I_rho=I_rho, Fx2=Fx2, Mx2=Mx2, My1=My1, Mz1=Mz1, My2=My2, Mz2=Mz2)
        k_g_global = T.T @ k_g_local @ T
        for a_idx in range(12):
            ga = int(dof_map[a_idx])
            for b_idx in range(12):
                gb = int(dof_map[b_idx])
                K[ga, gb] += k_g_global[a_idx, b_idx]
    return K