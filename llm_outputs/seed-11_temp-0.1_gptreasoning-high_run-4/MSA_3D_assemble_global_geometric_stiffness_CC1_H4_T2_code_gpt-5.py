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
    dof_per_node = 6
    n_dof = dof_per_node * n_nodes
    if u_global.shape[0] != n_dof:
        raise ValueError('Length of u_global does not match 6 * n_nodes.')
    K = np.zeros((n_dof, n_dof), dtype=float)
    g_z = np.array([0.0, 0.0, 1.0], dtype=float)
    g_y = np.array([0.0, 1.0, 0.0], dtype=float)
    tol = 1e-12
    for el in elements:
        i = int(el['node_i'])
        j = int(el['node_j'])
        xi = node_coords[i].astype(float)
        xj = node_coords[j].astype(float)
        dx = xj - xi
        L = float(np.linalg.norm(dx))
        if not np.isfinite(L) or L <= 0.0:
            raise ValueError('Element has zero or invalid length.')
        ex = dx / L
        ref = el.get('local_z', None)
        if ref is None:
            if abs(float(np.dot(ex, g_z))) >= 1.0 - 1e-08:
                ref_vec = g_y.copy()
            else:
                ref_vec = g_z.copy()
        else:
            ref_arr = np.asarray(ref, dtype=float).reshape(3)
            if not np.all(np.isfinite(ref_arr)):
                raise ValueError('Provided local_z contains non-finite values.')
            nrm = float(np.linalg.norm(ref_arr))
            if abs(nrm - 1.0) > 1e-06:
                raise ValueError('Provided local_z must be unit length.')
            if abs(float(np.dot(ex, ref_arr))) >= 1.0 - 1e-08:
                raise ValueError('Provided local_z is parallel to the element axis.')
            ref_vec = ref_arr
        proj = float(np.dot(ref_vec, ex))
        ez = ref_vec - proj * ex
        nz = float(np.linalg.norm(ez))
        if nz <= tol:
            raise ValueError('Invalid reference vector leads to undefined local axes.')
        ez = ez / nz
        ey = np.cross(ez, ex)
        ny = float(np.linalg.norm(ey))
        if ny <= tol:
            raise ValueError('Local axis construction failed (degenerate ey).')
        ey = ey / ny
        R = np.vstack((ex, ey, ez))
        T = np.zeros((12, 12), dtype=float)
        T[0:3, 0:3] = R
        T[3:6, 3:6] = R
        T[6:9, 6:9] = R
        T[9:12, 9:12] = R
        g_dofs = np.array([i * dof_per_node + 0, i * dof_per_node + 1, i * dof_per_node + 2, i * dof_per_node + 3, i * dof_per_node + 4, i * dof_per_node + 5, j * dof_per_node + 0, j * dof_per_node + 1, j * dof_per_node + 2, j * dof_per_node + 3, j * dof_per_node + 4, j * dof_per_node + 5], dtype=int)
        u_e_global = u_global[g_dofs]
        u_local = T @ u_e_global
        E = float(el['E'])
        nu = float(el['nu'])
        A = float(el['A'])
        Iy = float(el['I_y'])
        Iz = float(el['I_z'])
        J = float(el['J'])
        if A <= 0.0 or E <= 0.0 or Iy < 0.0 or (Iz < 0.0) or (J < 0.0):
            raise ValueError('Invalid section or material properties.')
        G = E / (2.0 * (1.0 + nu))
        k_e = np.zeros((12, 12), dtype=float)
        EA_L = E * A / L
        k_e[0, 0] += EA_L
        k_e[0, 6] += -EA_L
        k_e[6, 0] += -EA_L
        k_e[6, 6] += EA_L
        GJ_L = G * J / L
        k_e[3, 3] += GJ_L
        k_e[3, 9] += -GJ_L
        k_e[9, 3] += -GJ_L
        k_e[9, 9] += GJ_L
        EIz = E * Iz
        L2 = L * L
        L3 = L2 * L
        c1 = 12.0 * EIz / L3
        c2 = 6.0 * EIz / L2
        c3 = 4.0 * EIz / L
        c4 = 2.0 * EIz / L
        idx_vz = [1, 5, 7, 11]
        k_bz = np.array([[c1, c2, -c1, c2], [c2, c3, -c2, c4], [-c1, -c2, c1, -c2], [c2, c4, -c2, c3]], dtype=float)
        for a in range(4):
            for b in range(4):
                k_e[idx_vz[a], idx_vz[b]] += k_bz[a, b]
        EIy = E * Iy
        c1y = 12.0 * EIy / L3
        c2y = 6.0 * EIy / L2
        c3y = 4.0 * EIy / L
        c4y = 2.0 * EIy / L
        idx_by = [2, 4, 8, 10]
        k_by = np.array([[c1y, -c2y, -c1y, -c2y], [-c2y, c3y, c2y, c4y], [-c1y, c2y, c1y, c2y], [-c2y, c4y, c2y, c3y]], dtype=float)
        for a in range(4):
            for b in range(4):
                k_e[idx_by[a], idx_by[b]] += k_by[a, b]
        f_local = k_e @ u_local
        Fx2 = float(f_local[6])
        Mx2 = float(f_local[9])
        My1 = float(f_local[4])
        Mz1 = float(f_local[5])
        My2 = float(f_local[10])
        Mz2 = float(f_local[11])
        I_rho = Iy + Iz
        k_g_local = local_geometric_stiffness_matrix_3D_beam(L=L, A=A, I_rho=I_rho, Fx2=Fx2, Mx2=Mx2, My1=My1, Mz1=Mz1, My2=My2, Mz2=Mz2)
        k_g_global = T.T @ k_g_local @ T
        for a in range(12):
            ga = g_dofs[a]
            for b in range(12):
                gb = g_dofs[b]
                K[ga, gb] += k_g_global[a, b]
    return K