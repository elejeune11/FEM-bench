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
    node_coords = np.asarray(node_coords, dtype=float)
    if node_coords.ndim != 2 or node_coords.shape[1] != 3:
        raise ValueError('node_coords must be (n_nodes,3)')
    n_nodes = node_coords.shape[0]
    if u_global is None:
        raise ValueError('u_global is required')
    u_global = np.asarray(u_global, dtype=float)
    if u_global.size != 6 * n_nodes:
        raise ValueError('u_global length mismatch with node_coords')
    K = np.zeros((6 * n_nodes, 6 * n_nodes), dtype=float)

    def _get_nodes_from_element(el: dict):
        for key in ('nodes', 'node_ids', 'connectivity', 'node_indices', 'node_index', 'node_idx', 'nodes_idx'):
            if key in el:
                pair = el[key]
                return tuple(pair)
        if 'i' in el and 'j' in el:
            return (el['i'], el['j'])
        raise KeyError("Element connectivity not found (expected key 'nodes' etc.)")
    tol = 1e-12
    for el in elements:
        nodes_pair = _get_nodes_from_element(el)
        if len(nodes_pair) != 2:
            raise ValueError('Element connectivity must be two node indices')
        (n1, n2) = (int(nodes_pair[0]), int(nodes_pair[1]))
        if not (0 <= n1 < n_nodes and 0 <= n2 < n_nodes):
            raise IndexError('Element node index out of range')
        x1 = node_coords[n1, :].astype(float)
        x2 = node_coords[n2, :].astype(float)
        dx = x2 - x1
        L = np.linalg.norm(dx)
        if L <= tol:
            raise ValueError('Zero-length element encountered')
        ex = dx / L
        local_z = el.get('local_z', None)
        if local_z is None:
            gz = np.array([0.0, 0.0, 1.0])
            if abs(np.dot(ex, gz)) > 1.0 - 1e-12:
                gz = np.array([0.0, 1.0, 0.0])
            zref = gz
        else:
            zref = np.asarray(local_z, dtype=float)
            if zref.shape != (3,):
                raise ValueError('local_z must be length-3')
            znorm = np.linalg.norm(zref)
            if znorm <= tol:
                raise ValueError('local_z must be non-zero')
            zref = zref / znorm
            if abs(np.dot(ex, zref)) > 1.0 - 1e-12:
                raise ValueError('local_z is parallel to beam axis')
        ey = np.cross(zref, ex)
        ey_norm = np.linalg.norm(ey)
        if ey_norm <= tol:
            raise ValueError('Invalid local_z: produced zero local y-axis')
        ey = ey / ey_norm
        ez = np.cross(ex, ey)
        ez_norm = np.linalg.norm(ez)
        if ez_norm <= tol:
            raise ValueError('Failed to construct orthonormal local basis')
        ez = ez / ez_norm
        R = np.column_stack((ex, ey, ez))
        T_node = np.zeros((6, 6), dtype=float)
        T_node[0:3, 0:3] = R.T
        T_node[3:6, 3:6] = R.T
        T_elem = np.zeros((12, 12), dtype=float)
        T_elem[0:6, 0:6] = T_node
        T_elem[6:12, 6:12] = T_node
        E = float(el.get('E', 0.0))
        A = float(el.get('A', 0.0))
        if E <= 0 or A <= 0:
            pass
        dof_indices = []
        base1 = 6 * n1
        base2 = 6 * n2
        dof_indices.extend(range(base1, base1 + 6))
        dof_indices.extend(range(base2, base2 + 6))
        u_e_global = u_global[dof_indices].astype(float)
        u_e_local = T_elem @ u_e_global
        u1_axial = u_e_local[0]
        u2_axial = u_e_local[6]
        if E > 0 and A > 0:
            N = E * A / L * (u2_axial - u1_axial)
        else:
            N = 0.0
        k_local = np.zeros((12, 12), dtype=float)
        P_over_L = N / L
        yy_block = P_over_L * np.array([[1.0, -1.0], [-1.0, 1.0]], dtype=float)
        idx_v = (1, 7)
        k_local[idx_v[0], idx_v[0]] += yy_block[0, 0]
        k_local[idx_v[0], idx_v[1]] += yy_block[0, 1]
        k_local[idx_v[1], idx_v[0]] += yy_block[1, 0]
        k_local[idx_v[1], idx_v[1]] += yy_block[1, 1]
        idx_w = (2, 8)
        k_local[idx_w[0], idx_w[0]] += yy_block[0, 0]
        k_local[idx_w[0], idx_w[1]] += yy_block[0, 1]
        k_local[idx_w[1], idx_w[0]] += yy_block[1, 0]
        k_local[idx_w[1], idx_w[1]] += yy_block[1, 1]
        k_rot_coeff = N * L / 6.0
        k_local[1, 5] += -k_rot_coeff / L if L != 0 else 0.0
        k_local[1, 11] += k_rot_coeff / L if L != 0 else 0.0
        k_local[7, 5] += k_rot_coeff / L if L != 0 else 0.0
        k_local[7, 11] += -k_rot_coeff / L if L != 0 else 0.0
        k_local[5, 1] = k_local[1, 5]
        k_local[11, 1] = k_local[1, 11]
        k_local[5, 7] = k_local[7, 5]
        k_local[11, 7] = k_local[7, 11]
        k_local = 0.5 * (k_local + k_local.T)
        K_elem_global = T_elem.T @ k_local @ T_elem
        for (a_local, a_global) in enumerate(dof_indices):
            for (b_local, b_global) in enumerate(dof_indices):
                K[a_global, b_global] += K_elem_global[a_local, b_local]
    K = 0.5 * (K + K.T)
    return K