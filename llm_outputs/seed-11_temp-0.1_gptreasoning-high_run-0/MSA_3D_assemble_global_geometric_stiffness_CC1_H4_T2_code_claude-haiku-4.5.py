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
    n_nodes = node_coords.shape[0]
    n_dof = 6 * n_nodes
    K_g = np.zeros((n_dof, n_dof))
    for (elem_idx, elem) in enumerate(elements):
        E = elem['E']
        nu = elem['nu']
        A = elem['A']
        I_y = elem['I_y']
        I_z = elem['I_z']
        J = elem['J']
        node_i = elem['node_i']
        node_j = elem['node_j']
        coord_i = node_coords[node_i]
        coord_j = node_coords[node_j]
        beam_axis = coord_j - coord_i
        L = np.linalg.norm(beam_axis)
        if L < 1e-14:
            continue
        local_x = beam_axis / L
        if 'local_z' in elem and elem['local_z'] is not None:
            local_z = np.array(elem['local_z'], dtype=float)
            local_z = local_z / np.linalg.norm(local_z)
        else:
            ref_z = np.array([0.0, 0.0, 1.0])
            if abs(abs(np.dot(local_x, ref_z)) - 1.0) < 1e-10:
                ref_z = np.array([0.0, 1.0, 0.0])
            local_z = ref_z - np.dot(ref_z, local_x) * local_x
            local_z = local_z / np.linalg.norm(local_z)
        local_y = np.cross(local_z, local_x)
        local_y = local_y / np.linalg.norm(local_y)
        R = np.array([local_x, local_y, local_z]).T
        Gamma = np.zeros((12, 12))
        for i in range(4):
            Gamma[3 * i:3 * i + 3, 3 * i:3 * i + 3] = R
            Gamma[3 * i + 6:3 * i + 9, 3 * i + 6:3 * i + 9] = R
        dof_i = np.array([6 * node_i, 6 * node_i + 1, 6 * node_i + 2, 6 * node_i + 3, 6 * node_i + 4, 6 * node_i + 5])
        dof_j = np.array([6 * node_j, 6 * node_j + 1, 6 * node_j + 2, 6 * node_j + 3, 6 * node_j + 4, 6 * node_j + 5])
        dof_indices = np.concatenate([dof_i, dof_j])
        u_global_elem = u_global[dof_indices]
        u_local = Gamma.T @ u_global_elem
        G = E / (2.0 * (1.0 + nu))
        I_rho = J
        k_e_local = np.zeros((12, 12))
        k_e_local[0, 0] = E * A / L
        k_e_local[0, 6] = -E * A / L
        k_e_local[6, 0] = -E * A / L
        k_e_local[6, 6] = E * A / L
        k_e_local[2, 2] = 12.0 * E * I_z / L ** 3
        k_e_local[2, 5] = 6.0 * E * I_z / L ** 2
        k_e_local[2, 8] = -12.0 * E * I_z / L ** 3
        k_e_local[2, 11] = 6.0 * E * I_z / L ** 2
        k_e_local[5, 2] = 6.0 * E * I_z / L ** 2
        k_e_local[5, 5] = 4.0 * E * I_z / L
        k_e_local[5, 8] = -6.0 * E * I_z / L ** 2
        k_e_local[5, 11] = 2.0 * E * I_z / L
        k_e_local[8, 2] = -12.0 * E * I_z / L ** 3
        k_e_local[8, 5] = -6.0 * E * I_z / L ** 2
        k_e_local[8, 8] = 12.0 * E * I_z / L ** 3
        k_e_local[8, 11] = -6.0 * E * I_z / L ** 2
        k_e_local[11, 2] = 6.0 * E * I_z / L ** 2
        k_e_local[11, 5] = 2.0 * E * I_z / L
        k_e_local[11, 8] = -6.0 * E * I_z / L ** 2
        k_e_local[11, 11] = 4.0 * E * I_z / L
        k_e_local[1, 1] = 12.0 * E * I_y / L ** 3
        k_e_local[1, 4] = -6.0 * E * I_y / L ** 2
        k_e_local[1, 7] = -12.0 * E * I_y / L ** 3
        k_e_local[1, 10] = -6.0 * E * I_y / L ** 2
        k_e_local[4, 1] = -6.0 * E * I_y / L ** 2
        k_e_local[4, 4] = 4.0 * E * I_y / L
        k_e_local[4, 7] = 6.0 * E * I_y / L ** 2
        k_e_local[4, 10] = 2.0 * E * I_y / L
        k_e_local[7, 1] = -12.0 * E * I_y / L ** 3
        k_e_local[7, 4] = 6.0 * E * I_y / L ** 2
        k_e_local[7, 7] = 12.0 * E * I_y / L ** 3
        k_e_local[7, 10] = 6.0 * E * I_y / L ** 2
        k_e_local[10, 1] = -6.0 * E * I_y / L ** 2
        k_e_local[10, 4] = 2.0 * E * I_y / L
        k_e_local[10, 7] = 6.0 * E * I_y / L ** 2
        k_e_local[10, 10] = 4.0 * E * I_y / L
        k_e_local[3, 3] = G * J / L
        k_e_local[3, 9] = -G * J / L
        k_e_local[9, 3] = -G * J / L
        k_e_local[9, 9] = G * J / L
        f_local = k_e_local @ u_local
        Fx2 = f_local[6]
        Mx2 = f_local[9]
        My1 = f_local[4]
        Mz1 = f_local[5]
        My2 = f_local[10]
        Mz2 = f_local[11]
        k_g_local = local_geometric_stiffness_matrix_3D_beam(L, A, I_rho, Fx2, Mx2, My1, Mz1, My2, Mz2)
        k_g_global = Gamma @ k_g_local @ Gamma.T
        for i in range(12):
            for j in range(12):
                K_g[dof_indices[i], dof_indices[j]] += k_g_global[i, j]
    return K_g