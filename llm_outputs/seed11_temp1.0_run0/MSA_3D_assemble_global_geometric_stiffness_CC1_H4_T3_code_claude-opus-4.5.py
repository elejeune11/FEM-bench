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
    n_nodes = node_coords.shape[0]
    n_dof = 6 * n_nodes
    K_g = np.zeros((n_dof, n_dof), dtype=float)
    for elem in elements:
        node_i = elem['node_i']
        node_j = elem['node_j']
        coord_i = node_coords[node_i]
        coord_j = node_coords[node_j]
        delta = coord_j - coord_i
        L = np.linalg.norm(delta)
        if L < 1e-12:
            raise ValueError('Zero-length element detected')
        local_x = delta / L
        local_z_ref = elem.get('local_z', None)
        if local_z_ref is None:
            global_z = np.array([0.0, 0.0, 1.0])
            global_y = np.array([0.0, 1.0, 0.0])
            if abs(np.dot(local_x, global_z)) > 0.999:
                local_z_ref = global_y
            else:
                local_z_ref = global_z
        local_z_ref = np.asarray(local_z_ref, dtype=float)
        local_y = np.cross(local_z_ref, local_x)
        local_y_norm = np.linalg.norm(local_y)
        if local_y_norm < 1e-12:
            raise ValueError('local_z is parallel to beam axis')
        local_y = local_y / local_y_norm
        local_z = np.cross(local_x, local_y)
        R = np.array([local_x, local_y, local_z])
        Gamma = np.zeros((12, 12), dtype=float)
        for k in range(4):
            Gamma[3 * k:3 * k + 3, 3 * k:3 * k + 3] = R
        E = elem['E']
        A = elem['A']
        I_y = elem['I_y']
        I_z = elem['I_z']
        J = elem['J']
        nu = elem['nu']
        G = E / (2.0 * (1.0 + nu))
        k_e_local = np.zeros((12, 12), dtype=float)
        k_e_local[0, 0] = E * A / L
        k_e_local[0, 6] = -E * A / L
        k_e_local[6, 0] = -E * A / L
        k_e_local[6, 6] = E * A / L
        k_e_local[3, 3] = G * J / L
        k_e_local[3, 9] = -G * J / L
        k_e_local[9, 3] = -G * J / L
        k_e_local[9, 9] = G * J / L
        k_e_local[1, 1] = 12 * E * I_z / L ** 3
        k_e_local[1, 5] = 6 * E * I_z / L ** 2
        k_e_local[1, 7] = -12 * E * I_z / L ** 3
        k_e_local[1, 11] = 6 * E * I_z / L ** 2
        k_e_local[5, 1] = 6 * E * I_z / L ** 2
        k_e_local[5, 5] = 4 * E * I_z / L
        k_e_local[5, 7] = -6 * E * I_z / L ** 2
        k_e_local[5, 11] = 2 * E * I_z / L
        k_e_local[7, 1] = -12 * E * I_z / L ** 3
        k_e_local[7, 5] = -6 * E * I_z / L ** 2
        k_e_local[7, 7] = 12 * E * I_z / L ** 3
        k_e_local[7, 11] = -6 * E * I_z / L ** 2
        k_e_local[11, 1] = 6 * E * I_z / L ** 2
        k_e_local[11, 5] = 2 * E * I_z / L
        k_e_local[11, 7] = -6 * E * I_z / L ** 2
        k_e_local[11, 11] = 4 * E * I_z / L
        k_e_local[2, 2] = 12 * E * I_y / L ** 3
        k_e_local[2, 4] = -6 * E * I_y / L ** 2
        k_e_local[2, 8] = -12 * E * I_y / L ** 3
        k_e_local[2, 10] = -6 * E * I_y / L ** 2
        k_e_local[4, 2] = -6 * E * I_y / L ** 2
        k_e_local[4, 4] = 4 * E * I_y / L
        k_e_local[4, 8] = 6 * E * I_y / L ** 2
        k_e_local[4, 10] = 2 * E * I_y / L
        k_e_local[8, 2] = -12 * E * I_y / L ** 3
        k_e_local[8, 4] = 6 * E * I_y / L ** 2
        k_e_local[8, 8] = 12 * E * I_y / L ** 3
        k_e_local[8, 10] = 6 * E * I_y / L ** 2
        k_e_local[10, 2] = -6 * E * I_y / L ** 2
        k_e_local[10, 4] = 2 * E * I_y / L
        k_e_local[10, 8] = 6 * E * I_y / L ** 2
        k_e_local[10, 10] = 4 * E * I_y / L
        dof_i = [6 * node_i + k for k in range(6)]
        dof_j = [6 * node_j + k for k in range(6)]
        elem_dofs = dof_i + dof_j
        u_elem_global = u_global[elem_dofs]
        u_elem_local = Gamma @ u_elem_global
        f_local = k_e_local @ u_elem_local
        P = f_local[6]
        Mx_j = f_local[9]
        My_i = f_local[4]
        My_j = f_local[10]
        Mz_i = f_local[5]
        Mz_j = f_local[11]
        k_g_local = np.zeros((12, 12), dtype=float)
        a = P / L
        b = 6 * P / (5 * L)
        c = P / 10
        d = 2 * P * L / 15
        e = -P * L / 30
        k_g_local[1, 1] = b
        k_g_local[1, 5] = c
        k_g_local[1, 7] = -b
        k_g_local[1, 11] = c
        k_g_local[5, 1] = c
        k_g_local[5, 5] = d
        k_g_local[5, 7] = -c
        k_g_local[5, 11] = e
        k_g_local[7, 1] = -b
        k_g_local[7, 5] = -c
        k_g_local[7, 7] = b
        k_g_local[7, 11] = -c
        k_g_local[11, 1] = c
        k_g_local[11, 5] = e
        k_g_local[11, 7] = -c
        k_g_local[11, 11] = d
        k_g_local[2, 2] = b
        k_g_local[2, 4] = -c
        k_g_local[2, 8] = -b
        k_g_local[2, 10] = -c
        k_g_local[4, 2] = -c
        k_g_local[4, 4] = d
        k_g_local[4, 8] = c
        k_g_local[4, 10] = e
        k_g_local[8, 2] = -b
        k_g_local[8, 4] = c
        k_g_local[8, 8] = b
        k_g_local[8, 10] = c
        k_g_local[10, 2] = -c
        k_g_local[10, 4] = e
        k_g_local[10, 8] = c
        k_g_local[10, 10] = d
        k_g_global = Gamma.T @ k_g_local @ Gamma
        for (ii, gi) in enumerate(elem_dofs):
            for (jj, gj) in enumerate(elem_dofs):
                K_g[gi, gj] += k_g_global[ii, jj]
    return K_g