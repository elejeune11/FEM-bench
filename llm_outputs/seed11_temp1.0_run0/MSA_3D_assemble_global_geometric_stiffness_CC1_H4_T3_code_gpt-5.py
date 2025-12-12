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
    if not isinstance(node_coords, np.ndarray) or node_coords.ndim != 2 or node_coords.shape[1] != 3:
        raise ValueError('node_coords must be an (n_nodes, 3) ndarray.')
    n_nodes = node_coords.shape[0]
    dof_total = 6 * n_nodes
    u_global = np.asarray(u_global, dtype=float)
    if u_global.ndim != 1 or u_global.size != dof_total:
        raise ValueError('u_global must be a 1D array of length 6*n_nodes.')

    def _get_element_nodes(ed: dict):
        if 'nodes' in ed:
            nodes = ed['nodes']
        elif 'node_indices' in ed:
            nodes = ed['node_indices']
        elif 'connectivity' in ed:
            nodes = ed['connectivity']
        else:
            candidates = [('n1', 'n2'), ('i', 'j'), ('node_i', 'node_j'), ('node1', 'node2'), ('start', 'end'), ('start_node', 'end_node'), ('from', 'to')]
            nodes = None
            for (a, b) in candidates:
                if a in ed and b in ed:
                    nodes = (ed[a], ed[b])
                    break
            if nodes is None:
                raise ValueError("Element connectivity not found. Provide 'nodes' or equivalent two-node keys.")
        nodes = tuple((int(x) for x in nodes))
        if len(nodes) != 2:
            raise ValueError('Each element must connect exactly two nodes.')
        (i, j) = nodes
        if not (0 <= i < n_nodes and 0 <= j < n_nodes):
            raise ValueError('Element node indices out of range.')
        if i == j:
            raise ValueError('Zero-length element (identical node indices).')
        return (i, j)

    def _build_rotation_and_length(pi: np.ndarray, pj: np.ndarray, local_z_ref):
        v = pj - pi
        L = float(np.linalg.norm(v))
        if L <= 0.0 or not np.isfinite(L):
            raise ValueError('Zero-length or invalid element length.')
        x_axis = v / L
        if local_z_ref is None:
            ref = np.array([0.0, 0.0, 1.0], dtype=float)
            if abs(np.dot(ref, x_axis)) > 1.0 - 1e-08:
                ref = np.array([0.0, 1.0, 0.0], dtype=float)
            z_ref = ref
        else:
            z_ref = np.asarray(local_z_ref, dtype=float).reshape(3)
            nz = float(np.linalg.norm(z_ref))
            if not np.isfinite(nz) or nz <= 0:
                raise ValueError('Provided local_z has invalid norm.')
            if abs(nz - 1.0) > 1e-08:
                raise ValueError('Provided local_z must be unit length.')
            if abs(np.dot(z_ref, x_axis)) > 1.0 - 1e-08:
                raise ValueError('Provided local_z is parallel to the element axis.')
        y_temp = np.cross(z_ref, x_axis)
        ny = float(np.linalg.norm(y_temp))
        if ny <= 1e-12:
            raise ValueError('Invalid reference vector for local axes (parallel to axis).')
        y_axis = y_temp / ny
        z_axis = np.cross(x_axis, y_axis)
        y_axis = y_axis / np.linalg.norm(y_axis)
        z_axis = z_axis / np.linalg.norm(z_axis)
        R = np.vstack((x_axis, y_axis, z_axis))
        return (R, L)

    def _Ke_local(E, nu, A, Iy, Iz, J, L):
        G = E / (2.0 * (1.0 + nu))
        Ke = np.zeros((12, 12), dtype=float)
        k_ax = E * A / L
        Ke[0, 0] += k_ax
        Ke[0, 6] -= k_ax
        Ke[6, 0] -= k_ax
        Ke[6, 6] += k_ax
        k_tor = G * J / L
        Ke[3, 3] += k_tor
        Ke[3, 9] -= k_tor
        Ke[9, 3] -= k_tor
        Ke[9, 9] += k_tor
        EIz = E * Iz
        L2 = L * L
        L3 = L2 * L
        kbz = EIz / L3
        idx_bz = [1, 5, 7, 11]
        M_bz = np.array([[12.0, 6.0 * L, -12.0, 6.0 * L], [6.0 * L, 4.0 * L2, -6.0 * L, 2.0 * L2], [-12.0, -6.0 * L, 12.0, -6.0 * L], [6.0 * L, 2.0 * L2, -6.0 * L, 4.0 * L2]], dtype=float) * kbz
        for a in range(4):
            ia = idx_bz[a]
            for b in range(4):
                ib = idx_bz[b]
                Ke[ia, ib] += M_bz[a, b]
        EIy = E * Iy
        kby = EIy / L3
        idx_by = [2, 4, 8, 10]
        M_by = np.array([[12.0, -6.0 * L, -12.0, -6.0 * L], [-6.0 * L, 4.0 * L2, 6.0 * L, 2.0 * L2], [-12.0, 6.0 * L, 12.0, 6.0 * L], [-6.0 * L, 2.0 * L2, 6.0 * L, 4.0 * L2]], dtype=float) * kby
        for a in range(4):
            ia = idx_by[a]
            for b in range(4):
                ib = idx_by[b]
                Ke[ia, ib] += M_by[a, b]
        return Ke

    def _Kg_local_axial(P, L):
        Kg = np.zeros((12, 12), dtype=float)
        c = P / (30.0 * L)
        G4 = np.array([[36.0, 3.0 * L, -36.0, 3.0 * L], [3.0 * L, 4.0 * L * L, -3.0 * L, -1.0 * L * L], [-36.0, -3.0 * L, 36.0, -3.0 * L], [3.0 * L, -1.0 * L * L, -3.0 * L, 4.0 * L * L]], dtype=float) * c
        idx_bz = [1, 5, 7, 11]
        for a in range(4):
            ia = idx_bz[a]
            for b in range(4):
                ib = idx_bz[b]
                Kg[ia, ib] += G4[a, b]
        idx_by = [2, 4, 8, 10]
        for a in range(4):
            ia = idx_by[a]
            for b in range(4):
                ib = idx_by[b]
                Kg[ia, ib] += G4[a, b]
        return Kg
    K_global = np.zeros((dof_total, dof_total), dtype=float)
    for elem in elements:
        (i, j) = _get_element_nodes(elem)
        pi = node_coords[i]
        pj = node_coords[j]
        local_z = elem.get('local_z', None) if isinstance(elem, dict) else None
        (R, L) = _build_rotation_and_length(pi, pj, local_z)
        E = float(elem['E'])
        nu = float(elem['nu'])
        A = float(elem['A'])
        Iy = float(elem['I_y'])
        Iz = float(elem['I_z'])
        J = float(elem['J'])
        T = np.zeros((12, 12), dtype=float)
        T[0:3, 0:3] = R
        T[3:6, 3:6] = R
        T[6:9, 6:9] = R
        T[9:12, 9:12] = R
        ue_g = np.zeros(12, dtype=float)
        ue_g[0:6] = u_global[6 * i:6 * i + 6]
        ue_g[6:12] = u_global[6 * j:6 * j + 6]
        ue_l = T @ ue_g
        Ke_l = _Ke_local(E, nu, A, Iy, Iz, J, L)
        fe_l = Ke_l @ ue_l
        P = 0.5 * (fe_l[6] - fe_l[0])
        Kg_l = _Kg_local_axial(P, L)
        Kg_g = T.T @ Kg_l @ T
        dofs_i = np.arange(6 * i, 6 * i + 6)
        dofs_j = np.arange(6 * j, 6 * j + 6)
        dofs = np.concatenate((dofs_i, dofs_j))
        K_global[np.ix_(dofs, dofs)] += Kg_g
    return K_global