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
        node_i = elem['node_i']
        node_j = elem['node_j']
        coord_i = node_coords[node_i]
        coord_j = node_coords[node_j]
        delta = coord_j - coord_i
        L = np.linalg.norm(delta)
        if L < 1e-14:
            raise ValueError(f'Element {elem_idx} has zero length')
        local_x = delta / L
        local_z_ref = elem.get('local_z', None)
        if local_z_ref is None:
            if abs(local_x[2]) > 0.9:
                local_z_ref = np.array([0.0, 1.0, 0.0])
            else:
                local_z_ref = np.array([0.0, 0.0, 1.0])
        else:
            local_z_ref = np.array(local_z_ref, dtype=float)
        local_y = np.cross(local_z_ref, local_x)
        norm_y = np.linalg.norm(local_y)
        if norm_y < 1e-10:
            raise ValueError(f'Element {elem_idx}: local_z is parallel to beam axis')
        local_y = local_y / norm_y
        local_z = np.cross(local_x, local_y)
        local_z = local_z / np.linalg.norm(local_z)
        R = np.array([local_x, local_y, local_z])
        Gamma = np.zeros((12, 12))
        Gamma[0:3, 0:3] = R
        Gamma[3:6, 3:6] = R
        Gamma[6:9, 6:9] = R
        Gamma[9:12, 9:12] = R
        E = elem['E']
        nu = elem['nu']
        A = elem['A']
        I_y = elem['I_y']
        I_z = elem['I_z']
        J = elem['J']
        G = E / (2.0 * (1.0 + nu))
        I_rho = I_y + I_z
        k_e_local = np.zeros((12, 12))
        EA_L = E * A / L
        k_e_local[0, 0] = EA_L
        k_e_local[0, 6] = -EA_L
        k_e_local[6, 0] = -EA_L
        k_e_local[6, 6] = EA_L
        GJ_L = G * J / L
        k_e_local[3, 3] = GJ_L
        k_e_local[3, 9] = -GJ_L
        k_e_local[9, 3] = -GJ_L
        k_e_local[9, 9] = GJ_L
        EIz = E * I_z
        k_e_local[1, 1] = 12 * EIz / L ** 3
        k_e_local[1, 5] = 6 * EIz / L ** 2
        k_e_local[1, 7] = -12 * EIz / L ** 3
        k_e_local[1, 11] = 6 * EIz / L ** 2
        k_e_local[5, 1] = 6 * EIz / L ** 2
        k_e_local[5, 5] = 4 * EIz / L
        k_e_local[5, 7] = -6 * EIz / L ** 2
        k_e_local[5, 11] = 2 * EIz / L
        k_e_local[7, 1] = -12 * EIz / L ** 3
        k_e_local[7, 5] = -6 * EIz / L ** 2
        k_e_local[7, 7] = 12 * EIz / L ** 3
        k_e_local[7, 11] = -6 * EIz / L ** 2
        k_e_local[11, 1] = 6 * EIz / L ** 2
        k_e_local[11, 5] = 2 * EIz / L
        k_e_local[11, 7] = -6 * EIz / L ** 2
        k_e_local[11, 11] = 4 * EIz / L
        EIy = E * I_y
        k_e_local[2, 2] = 12 * EIy / L ** 3
        k_e_local[2, 4] = -6 * EIy / L ** 2
        k_e_local[2, 8] = -12 * EIy / L ** 3
        k_e_local[2, 10] = -6 * EIy / L ** 2
        k_e_local[4, 2] = -6 * EIy / L ** 2
        k_e_local[4, 4] = 4 * EIy / L
        k_e_local[4, 8] = 6 * EIy / L ** 2
        k_e_local[4, 10] = 2 * EIy / L
        k_e_local[8, 2] = -12 * EIy / L ** 3
        k_e_local[8, 4] = 6 * EIy / L ** 2
        k_e_local[8, 8] = 12 * EIy / L ** 3
        k_e_local[8, 10] = 6 * EIy / L ** 2
        k_e_local[10, 2] = -6 * EIy / L ** 2
        k_e_local[10, 4] = 2 * EIy / L
        k_e_local[10, 8] = 6 * EIy / L ** 2
        k_e_local[10, 10] = 4 * EIy / L
        dof_i = np.arange(6 * node_i, 6 * node_i + 6)
        dof_j = np.arange(6 * node_j, 6 * node_j + 6)
        elem_dofs = np.concatenate([dof_i, dof_j])
        u_elem_global = u_global[elem_dofs]
        u_elem_local = Gamma @ u_elem_global
        f_elem_local = k_e_local @ u_elem_local
        Fx2 = f_elem_local[6]
        Mx2 = f_elem_local[9]
        My1 = f_elem_local[4]
        Mz1 = f_elem_local[5]
        My2 = f_elem_local[10]
        Mz2 = f_elem_local[11]
        k_g_local = local_geometric_stiffness_matrix_3D_beam(L, A, I_rho, Fx2, Mx2, My1, Mz1, My2, Mz2)
        k_g_global = Gamma.T @ k_g_local @ Gamma
        for ii in range(12):
            for jj in range(12):
                K_g[elem_dofs[ii], elem_dofs[jj]] += k_g_global[ii, jj]
    return K_g