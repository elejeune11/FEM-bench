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
    n_dofs = 6 * n_nodes
    K_g = np.zeros((n_dofs, n_dofs))
    for element in elements:
        (node_i, node_j) = (element['node_i'], element['node_j'])
        L = np.linalg.norm(node_coords[node_j] - node_coords[node_i])
        if L == 0:
            raise ValueError('Zero-length element encountered')
        if 'local_z' not in element:
            beam_axis = node_coords[node_j] - node_coords[node_i]
            if np.abs(beam_axis[2]) < 1e-06:
                element['local_z'] = np.array([0, 1, 0])
            else:
                element['local_z'] = np.array([0, 0, 1])
        beam_axis = (node_coords[node_j] - node_coords[node_i]) / L
        local_z = np.array(element['local_z'])
        local_y = np.cross(beam_axis, local_z)
        local_y /= np.linalg.norm(local_y)
        local_x = np.cross(local_y, local_z)
        T = np.array([local_x, local_y, local_z]).T
        Fx = element['E'] * element['A'] * (u_global[6 * node_j] - u_global[6 * node_i]) / L
        k_g_local = np.zeros((12, 12))
        k_g_local[0, 0] = Fx / L
        k_g_local[6, 6] = Fx / L
        k_g_global = T.T @ k_g_local @ T
        dof_i = np.arange(6 * node_i, 6 * node_i + 6)
        dof_j = np.arange(6 * node_j, 6 * node_j + 6)
        K_g[np.ix_(dof_i, dof_i)] += k_g_global[:6, :6]
        K_g[np.ix_(dof_i, dof_j)] += k_g_global[:6, 6:]
        K_g[np.ix_(dof_j, dof_i)] += k_g_global[6:, :6]
        K_g[np.ix_(dof_j, dof_j)] += k_g_global[6:, 6:]
    return K_g