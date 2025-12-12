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
        node_i = int(elem['node_i'])
        node_j = int(elem['node_j'])
        E = elem['E']
        nu = elem['nu']
        A = elem['A']
        I_y = elem['I_y']
        I_z = elem['I_z']
        J = elem['J']
        xi = node_coords[node_i]
        xj = node_coords[node_j]
        beam_axis = xj - xi
        L = np.linalg.norm(beam_axis)
        if L < 1e-12:
            continue
        beam_axis_unit = beam_axis / L
        if 'local_z' in elem and elem['local_z'] is not None:
            local_z = np.array(elem['local_z'], dtype=float)
            local_z = local_z / np.linalg.norm(local_z)
        else:
            global_z = np.array([0.0, 0.0, 1.0])
            global_y = np.array([0.0, 1.0, 0.0])
            dot_with_z = abs(np.dot(beam_axis_unit, global_z))
            if dot_with_z < 0.9:
                local_z = global_z - np.dot(global_z, beam_axis_unit) * beam_axis_unit
            else:
                local_z = global_y - np.dot(global_y, beam_axis_unit) * beam_axis_unit
            local_z = local_z / np.linalg.norm(local_z)
        local_x = beam_axis_unit
        local_y = np.cross(local_z, local_x)
        local_y = local_y / np.linalg.norm(local_y)
        R = np.array([local_x, local_y, local_z]).T
        Gamma = np.zeros((12, 12))
        for block in range(4):
            Gamma[3 * block:3 * block + 3, 3 * block:3 * block + 3] = R
        dof_i = np.arange(6 * node_i, 6 * node_i + 6)
        dof_j = np.arange(6 * node_j, 6 * node_j + 6)
        dof_elem = np.concatenate([dof_i, dof_j])
        u_local = Gamma.T @ u_global[dof_elem]
        G = E / (2.0 * (1.0 + nu))
        k_e = np.zeros((12, 12))
        k_e[0, 0] = E * A / L
        k_e[0, 6] = -E * A / L
        k_e[6, 0] = -E * A / L
        k_e[6, 6] = E * A / L
        k_e[1, 1] = 12.0 * E * I_z / L ** 3
        k_e[1, 5] = 6.0 * E * I_z / L ** 2
        k_e[1, 7] = -12.0 * E * I_z / L ** 3
        k_e[1, 11] = 6.0 * E * I_z / L ** 2
        k_e[5, 1] = 6.0 * E * I_z / L ** 2
        k_e[5, 5] = 4.0 * E * I_z / L
        k_e[5, 7] = -6.0 * E * I_z / L ** 2
        k_e[5, 11] = 2.0 * E * I_z / L
        k_e[7, 1] = -12.0 * E * I_z / L ** 3
        k_e[7, 5] = -6.0 * E * I_z / L ** 2
        k_e[7, 7] = 12.0 * E * I_z / L ** 3
        k_e[7, 11] = -6.0 * E * I_z / L ** 2
        k_e[11, 1] = 6.0 * E * I_z / L ** 2
        k_e[11, 5] = 2.0 * E * I_z / L
        k_e[11, 7] = -6.0 * E * I_z / L ** 2
        k_e[11, 11] = 4.0 * E * I_z / L
        k_e[2, 2] = 12.0 * E * I_y / L ** 3
        k_e[2, 4] = -6.0 * E * I_y / L ** 2
        k_e[2, 8] = -12.0 * E * I_y / L ** 3
        k_e[2, 10] = -6.0 * E * I_y / L ** 2
        k_e[4, 2] = -6.0 * E * I_y / L ** 2
        k_e[4, 4] = 4.0 * E * I_y / L
        k_e[4, 8] = 6.0 * E * I_y / L ** 2
        k_e[4, 10] = 2.0 * E * I_y / L
        k_e[8, 2] = -12.0 * E * I_y / L ** 3
        k_e[8, 4] = 6.0 * E * I_y / L ** 2
        k_e[8, 8] = 12.0 * E * I_y / L ** 3
        k_e[8, 10] = 6.0 * E * I_y / L ** 2
        k_e[10, 2] = -6.0 * E * I_y / L ** 2
        k_e[10, 4] = 2.0 * E * I_y / L
        k_e[10, 8] = 6.0 * E * I_y / L ** 2
        k_e[10, 10] = 4.0 * E * I_y / L
        k_e[3, 3] = G * J / L
        k_e[3, 9] = -G * J / L
        k_e[9, 3] = -G * J / L
        k_e[9, 9] = G * J / L
        f_local = k_e @ u_local
        (Fx1, Fy1, Fz1) = (f_local[0], f_local[1], f_local[2])
        (Mx1, My1, Mz1) = (f_local[3], f_local[4], f_local[5])
        (Fx2, Fy2, Fz2) = (f_local[6], f_local[7], f_local[8])
        (Mx2, My2, Mz2) = (f_local[9], f_local[10], f_local[11])
        I_rho = I_y + I_z
        k_g_local = local_geometric_stiffness_matrix_3D_beam(L, A, I_rho, Fx2, Mx2, My1, Mz1, My2, Mz2)
        k_g_global = Gamma @ k_g_local @ Gamma.T
        for i in range(12):
            for j in range(12):
                global_i = dof_elem[i]
                global_j = dof_elem[j]
                K_g[global_i, global_j] += k_g_global[i, j]
    return K_g