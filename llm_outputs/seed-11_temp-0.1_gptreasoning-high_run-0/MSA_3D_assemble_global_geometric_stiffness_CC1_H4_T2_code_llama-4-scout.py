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
    for element in elements:
        (node_i, node_j) = element['nodes']
        L = np.linalg.norm(node_coords[node_j] - node_coords[node_i])
        local_x = (node_coords[node_j] - node_coords[node_i]) / L
        if 'local_z' in element:
            local_z = np.array(element['local_z'])
        elif np.isclose(local_x, np.array([0, 0, 1])).all():
            local_z = np.array([0, 1, 0])
        else:
            local_z = np.array([0, 0, 1])
        local_z = local_z / np.linalg.norm(local_z)
        local_y = np.cross(local_x, local_z)
        local_y = local_y / np.linalg.norm(local_y)
        T = np.array([local_x, local_y, local_z]).T
        u_local_i = u_global[6 * node_i:6 * node_i + 6]
        u_local_j = u_global[6 * node_j:6 * node_j + 6]
        u_local = np.concatenate((u_local_i, u_local_j))
        E = element['E']
        A = element['A']
        I_y = element['I_y']
        I_z = element['I_z']
        J = element['J']
        Fx_i = E * A * (u_local[6] - u_local[0]) / L
        Fy_i = E * I_z * (u_local[4] - u_local[1]) / L ** 3
        Fz_i = E * I_y * (u_local[5] - u_local[2]) / L ** 3
        Mx_i = G * J * (u_local[11] - u_local[8]) / L
        My_i = E * I_z * (u_local[9] - u_local[3]) / L
        Mz_i = E * I_y * (u_local[10] - u_local[7]) / L
        Fx_j = -Fx_i
        Fy_j = -Fy_i
        Fz_j = -Fz_i
        Mx_j = G * J * (u_local[11] - u_local[8]) / L
        My_j = -My_i
        Mz_j = -Mz_i
        I_rho = J
        k_g_local = local_geometric_stiffness_matrix_3D_beam(L, A, I_rho, Fx_j, Mx_j, My_i, Mz_i, My_j, Mz_j)
        T_global = np.tile(T, (2, 1, 1)).reshape((12, 12))
        k_g_global = T_global @ k_g_local @ T_global.transpose()
        dof_i = np.arange(6 * node_i, 6 * node_i + 6)
        dof_j = np.arange(6 * node_j, 6 * node_j + 6)
        dof_map = np.concatenate((dof_i, dof_j))
        K_g[np.ix_(dof_map, dof_map)] += k_g_global
    return K_g