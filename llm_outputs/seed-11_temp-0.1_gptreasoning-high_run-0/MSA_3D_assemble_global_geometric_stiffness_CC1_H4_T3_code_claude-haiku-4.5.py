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
    n_nodes = node_coords.shape[0]
    n_dof = 6 * n_nodes
    K_g = np.zeros((n_dof, n_dof))
    for elem in elements:
        node_i = elem['node_i']
        node_j = elem['node_j']
        r_i = node_coords[node_i]
        r_j = node_coords[node_j]
        axis = r_j - r_i
        L = np.linalg.norm(axis)
        if L == 0:
            continue
        e_x = axis / L
        if 'local_z' in elem and elem['local_z'] is not None:
            ref_z = np.array(elem['local_z'], dtype=float)
        elif abs(e_x[2]) < 0.9:
            ref_z = np.array([0.0, 0.0, 1.0])
        else:
            ref_z = np.array([0.0, 1.0, 0.0])
        ref_z = ref_z / np.linalg.norm(ref_z)
        e_y = np.cross(ref_z, e_x)
        e_y = e_y / np.linalg.norm(e_y)
        e_z = np.cross(e_x, e_y)
        e_z = e_z / np.linalg.norm(e_z)
        T = np.array([e_x, e_y, e_z])
        Gamma = np.zeros((12, 12))
        for i in range(2):
            Gamma[6 * i:6 * i + 3, 6 * i:6 * i + 3] = T
            Gamma[6 * i + 3:6 * i + 6, 6 * i + 3:6 * i + 6] = T
        E = elem['E']
        nu = elem['nu']
        A = elem['A']
        I_y = elem['I_y']
        I_z = elem['I_z']
        J = elem['J']
        u_i_global = u_global[6 * node_i:6 * node_i + 6]
        u_j_global = u_global[6 * node_j:6 * node_j + 6]
        u_global_elem = np.concatenate([u_i_global, u_j_global])
        u_local = Gamma @ u_global_elem
        G = E / (2 * (1 + nu))
        k_e_local = np.zeros((12, 12))
        k_e_local[0, 0] = E * A / L
        k_e_local[0, 6] = -E * A / L
        k_e_local[6, 0] = -E * A / L
        k_e_local[6, 6] = E * A / L
        EI_z = E * I_z
        k_e_local[1, 1] = 12 * EI_z / L ** 3
        k_e_local[1, 5] = 6 * EI_z / L ** 2
        k_e_local[1, 7] = -12 * EI_z / L ** 3
        k_e_local[1, 11] = 6 * EI_z / L ** 2
        k_e_local[5, 1] = 6 * EI_z / L ** 2
        k_e_local[5, 5] = 4 * EI_z / L
        k_e_local[5, 7] = -6 * EI_z / L ** 2
        k_e_local[5, 11] = 2 * EI_z / L
        k_e_local[7, 1] = -12 * EI_z / L ** 3
        k_e_local[7, 5] = -6 * EI_z / L ** 2
        k_e_local[7, 7] = 12 * EI_z / L ** 3
        k_e_local[7, 11] = -6 * EI_z / L ** 2
        k_e_local[11, 1] = 6 * EI_z / L ** 2
        k_e_local[11, 5] = 2 * EI_z / L
        k_e_local[11, 7] = -6 * EI_z / L ** 2
        k_e_local[11, 11] = 4 * EI_z / L
        EI_y = E * I_y
        k_e_local[2, 2] = 12 * EI_y / L ** 3
        k_e_local[2, 4] = -6 * EI_y / L ** 2
        k_e_local[2, 8] = -12 * EI_y / L ** 3
        k_e_local[2, 10] = -6 * EI_y / L ** 2
        k_e_local[4, 2] = -6 * EI_y / L ** 2
        k_e_local[4, 4] = 4 * EI_y / L
        k_e_local[4, 8] = 6 * EI_y / L ** 2
        k_e_local[4, 10] = 2 * EI_y / L
        k_e_local[8, 2] = -12 * EI_y / L ** 3
        k_e_local[8, 4] = 6 * EI_y / L ** 2
        k_e_local[8, 8] = 12 * EI_y / L ** 3
        k_e_local[8, 10] = 6 * EI_y / L ** 2
        k_e_local[10, 2] = -6 * EI_y / L ** 2
        k_e_local[10, 4] = 2 * EI_y / L
        k_e_local[10, 8] = 6 * EI_y / L ** 2
        k_e_local[10, 10] = 4 * EI_y / L
        GJ = G * J
        k_e_local[3, 3] = GJ / L
        k_e_local[3, 9] = -GJ / L
        k_e_local[9, 3] = -GJ / L
        k_e_local[9, 9] = GJ / L
        f_local = k_e_local @ u_local
        Fx_i = f_local[0]
        Fy_i = f_local[1]
        Fz_i = f_local[2]
        Mx_i = f_local[3]
        My_i = f_local[4]
        Mz_i = f_local[5]
        Fx_j = f_local[6]
        Fy_j = f_local[7]
        Fz_j = f_local[8]
        Mx_j = f_local[9]
        My_j = f_local[10]
        Mz_j = f_local[11]
        k_g_local = np.zeros((12, 12))
        P = Fx_j
        if abs(P) > 1e-14:
            c1 = P / (30 * L)
            c2 = P * L / 840
            k_g_local[1, 1] += 6 * c1
            k_g_local[1, 5] += c2
            k_g_local[1, 7] -= 6 * c1
            k_g_local[1, 11] += c2
            k_g_local[5, 1] += c2
            k_g_local[5, 5] += P * L / 105
            k_g_local[5, 7] -= c2
            k_g_local[5, 11] -= P * L / 140
            k_g_local[7, 1] -= 6 * c1
            k_g_local[7, 5] -= c2
            k_g_local[7, 7] += 6 * c1
            k_g_local[7, 11] -= c2
            k_g_local[11, 1] += c2
            k_g_local[11, 5] -= P * L / 140
            k_g_local[11, 7] -= c2
            k_g_local[11, 11] += P * L / 105
            k_g_local[2, 2] += 6 * c1
            k_g_local[2, 4] -= c2
            k_g_local[2, 8] -= 6 * c1
            k_g_local[2, 10] -= c2
            k_g_local[4, 2] -= c2
            k_g_local[4, 4] += P * L / 105
            k_g_local[4, 8] += c2
            k_g_local[4, 10] -= P * L / 140
            k_g_local[8, 2] -= 6 * c1
            k_g_local[8, 4] += c2
            k_g_local[8, 8] += 6 * c1
            k_g_local[8, 10] += c2
            k_g_local[10, 2] -= c2
            k_g_local[10, 4] -= P * L / 140
            k_g_local[10, 8] += c2
            k_g_local[10, 10] += P * L / 105
        if abs(Mx_j) > 1e-14:
            c_mx = Mx_j / (30 * L)
            k_g_local[2, 4] -= c_mx
            k_g_local[4, 2] -= c_mx
            k_g_local[2, 10] += c_mx
            k_g_local[10, 2] += c_mx
            k_g_local[4, 8] += c_mx
            k_g_local[8, 4] += c_mx
            k_g_local[8, 10] -= c_mx
            k_g_local[10, 8] -= c_mx
        if abs(My_j) > 1e-14:
            c_my = My_j / (30 * L)
            k_g_local[1, 3] += c_my
            k_g_local[3, 1] += c_my
            k_g_local[1, 9] -= c_my
            k_g_local[9, 1] -= c_my
            k_g_local[3, 7] -= c_my
            k_g_local[7, 3] -= c_my
            k_g_local[7, 9] += c_my
            k_g_local[9, 7] += c_my
        if abs(Mz_j) > 1e-14:
            c_mz = Mz_j / (30 * L)
            k_g_local[2, 3] -= c_mz
            k_g_local[3, 2] -= c_mz
            k_g_local[2, 9] += c_mz
            k_g_local[9, 2] += c_mz
            k_g_local[3, 8] += c_mz
            k_g_local[8, 3] += c_mz
            k_g_local[8, 9] -= c_mz
            k_g_local[9, 8] -= c_mz
        k_g_global = Gamma.T @ k_g_local @ Gamma
        dof_i = np.arange(6 * node_i, 6 * node_i + 6)
        dof_j = np.arange(6 * node_j, 6 * node_j + 6)
        dof = np.concatenate([dof_i, dof_j])
        for (a, dof_a) in enumerate(dof):
            for (b, dof_b) in enumerate(dof):
                K_g[dof_a, dof_b] += k_g_global[a, b]
    return K_g