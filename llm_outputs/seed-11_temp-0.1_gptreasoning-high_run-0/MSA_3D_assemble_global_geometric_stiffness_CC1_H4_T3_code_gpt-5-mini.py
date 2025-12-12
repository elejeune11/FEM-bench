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
    tol = 1e-12
    node_coords = np.asarray(node_coords, dtype=float)
    if node_coords.ndim != 2 or node_coords.shape[1] != 3:
        pytest.fail('node_coords must be (n_nodes,3) array')
    n_nodes = node_coords.shape[0]
    u_global = np.asarray(u_global, dtype=float).ravel()
    if u_global.size != 6 * n_nodes:
        pytest.fail('u_global length must be 6*n_nodes')
    K = np.zeros((6 * n_nodes, 6 * n_nodes), dtype=float)
    for elem in elements:
        if not all((k in elem for k in ('node_i', 'node_j', 'E', 'A', 'I_y', 'I_z', 'J', 'nu'))):
            pytest.fail('Element dictionary missing required keys')
        ni = int(elem['node_i'])
        nj = int(elem['node_j'])
        if ni < 0 or nj < 0 or ni >= n_nodes or (nj >= n_nodes):
            pytest.fail('Element node index out of range')
        r_i = node_coords[ni, :].astype(float)
        r_j = node_coords[nj, :].astype(float)
        axis = r_j - r_i
        L = np.linalg.norm(axis)
        if L <= tol:
            pytest.fail('Zero-length element encountered')
        e_x = axis / L
        local_z_input = elem.get('local_z', None)
        if local_z_input is None:
            ref = np.array([0.0, 0.0, 1.0], dtype=float)
            if abs(np.dot(ref, e_x)) > 1.0 - 1e-08:
                ref = np.array([0.0, 1.0, 0.0], dtype=float)
            v_ref = ref
        else:
            v_ref = np.asarray(local_z_input, dtype=float)
            if v_ref.shape != (3,):
                pytest.fail('local_z must be a 3-vector')
            norm_v = np.linalg.norm(v_ref)
            if abs(norm_v - 1.0) > 1e-08:
                pytest.fail('local_z must be unit length')
            if abs(np.dot(v_ref, e_x)) > 1.0 - 1e-08:
                pytest.fail('local_z is parallel to the beam axis')
        e_y = np.cross(v_ref, e_x)
        n_e_y = np.linalg.norm(e_y)
        if n_e_y <= tol:
            pytest.fail('Invalid orientation: local_z is parallel to beam axis or numerical degeneracy')
        e_y = e_y / n_e_y
        e_z = np.cross(e_x, e_y)
        e_z = e_z / np.linalg.norm(e_z)
        R = np.column_stack((e_x, e_y, e_z))
        G6 = np.zeros((6, 6), dtype=float)
        G6[0:3, 0:3] = R
        G6[3:6, 3:6] = R
        T12 = np.zeros((12, 12), dtype=float)
        T12[0:6, 0:6] = G6
        T12[6:12, 6:12] = G6
        dof_indices = [6 * ni + i for i in range(6)] + [6 * nj + i for i in range(6)]
        u_e_global = np.zeros(12, dtype=float)
        u_e_global[0:6] = u_global[6 * ni:6 * ni + 6]
        u_e_global[6:12] = u_global[6 * nj:6 * nj + 6]
        u_local = T12.T @ u_e_global
        E = float(elem['E'])
        A = float(elem['A'])
        N = E * A / L * (u_local[6] - u_local[0])
        if abs(N) <= 0.0:
            continue
        k_g_local = np.zeros((12, 12), dtype=float)
        coeff = N / (30.0 * L)
        k_block = coeff * np.array([[36.0, 3.0 * L, -36.0, 3.0 * L], [3.0 * L, 4.0 * L * L, -3.0 * L, -1.0 * L * L], [-36.0, -3.0 * L, 36.0, -3.0 * L], [3.0 * L, -1.0 * L * L, -3.0 * L, 4.0 * L * L]], dtype=float)
        idx_v = [1, 5, 7, 11]
        idx_w = [2, 4, 8, 10]
        for a in range(4):
            for b in range(4):
                k_g_local[idx_v[a], idx_v[b]] += k_block[a, b]
                k_g_local[idx_w[a], idx_w[b]] += k_block[a, b]
        K_elem_global = T12 @ k_g_local @ T12.T
        ix = np.ix_(dof_indices, dof_indices)
        K[ix] += K_elem_global
    return K