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

    def _get_element_nodes(elem: dict) -> tuple[int, int]:
        for key in ('nodes', 'conn', 'connectivity', 'node_ids', 'indices'):
            if key in elem:
                nodes = elem[key]
                break
        else:
            raise KeyError("Element dictionary missing 'nodes' (or alias) connectivity key.")
        if len(nodes) != 2:
            raise ValueError('Each element must connect exactly two nodes.')
        (i, j) = (int(nodes[0]), int(nodes[1]))
        return (i, j)

    def _rotation_matrix_and_length(xi: np.ndarray, xj: np.ndarray, ref_z: Optional[np.ndarray]) -> tuple[np.ndarray, float]:
        dx = xj - xi
        L = float(np.linalg.norm(dx))
        if not np.isfinite(L) or L <= 0.0:
            raise ValueError('Zero or invalid element length.')
        ex = dx / L
        if ref_z is None:
            cand = np.array([0.0, 0.0, 1.0])
            if abs(np.dot(ex, cand)) > 1.0 - 1e-12:
                cand = np.array([0.0, 1.0, 0.0])
            ref = cand
        else:
            ref = np.asarray(ref_z, dtype=float).reshape(3)
            nrm = np.linalg.norm(ref)
            if not np.isfinite(nrm) or nrm <= 0.0:
                raise ValueError('Provided local_z must be a non-zero vector.')
            ref = ref / nrm
        cosang = abs(np.dot(ex, ref))
        if cosang > 1.0 - 1e-12:
            raise ValueError('Provided local_z is parallel (or nearly) to the beam axis.')
        ez_tmp = ref - np.dot(ref, ex) * ex
        ez_norm = np.linalg.norm(ez_tmp)
        if ez_norm <= 0.0:
            raise ValueError('Failed to construct local axes: degenerate reference vector.')
        ez = ez_tmp / ez_norm
        ey = np.cross(ez, ex)
        R = np.column_stack((ex, ey, ez))
        return (R, L)

    def _T_from_R(R: np.ndarray) -> np.ndarray:
        T = np.zeros((12, 12), dtype=float)
        T[0:3, 0:3] = R
        T[3:6, 3:6] = R
        T[6:9, 6:9] = R
        T[9:12, 9:12] = R
        return T

    def _local_linear_stiffness(E: float, G: float, A: float, Iy: float, Iz: float, J: float, L: float) -> np.ndarray:
        k = np.zeros((12, 12), dtype=float)
        k_ax = E * A / L
        k[0, 0] = k_ax
        k[0, 6] = -k_ax
        k[6, 0] = -k_ax
        k[6, 6] = k_ax
        k_tx = G * J / L
        k[3, 3] = k_tx
        k[3, 9] = -k_tx
        k[9, 3] = -k_tx
        k[9, 9] = k_tx
        EIz = E * Iz
        c = EIz / L ** 3
        idx = (1, 5, 7, 11)
        sub = np.array([[12.0, 6.0 * L, -12.0, 6.0 * L], [6.0 * L, 4.0 * L ** 2, -6.0 * L, 2.0 * L ** 2], [-12.0, -6.0 * L, 12.0, -6.0 * L], [6.0 * L, 2.0 * L ** 2, -6.0 * L, 4.0 * L ** 2]], dtype=float) * c
        for a in range(4):
            for b in range(4):
                k[idx[a], idx[b]] += sub[a, b]
        EIy = E * Iy
        c = EIy / L ** 3
        idx = (2, 4, 8, 10)
        sub = np.array([[12.0, -6.0 * L, -12.0, -6.0 * L], [-6.0 * L, 4.0 * L ** 2, 6.0 * L, 2.0 * L ** 2], [-12.0, 6.0 * L, 12.0, 6.0 * L], [-6.0 * L, 2.0 * L ** 2, 6.0 * L, 4.0 * L ** 2]], dtype=float) * c
        for a in range(4):
            for b in range(4):
                k[idx[a], idx[b]] += sub[a, b]
        return k
    n_nodes = int(node_coords.shape[0])
    if u_global.shape[0] != 6 * n_nodes:
        raise ValueError('u_global size incompatible with node count.')
    K = np.zeros((6 * n_nodes, 6 * n_nodes), dtype=float)
    for elem in elements:
        (i, j) = _get_element_nodes(elem)
        xi = np.asarray(node_coords[i], dtype=float).reshape(3)
        xj = np.asarray(node_coords[j], dtype=float).reshape(3)
        ref_z = elem.get('local_z', None)
        (R, L) = _rotation_matrix_and_length(xi, xj, ref_z)
        T = _T_from_R(R)
        E = float(elem['E'])
        nu = float(elem['nu'])
        A = float(elem['A'])
        Iy = float(elem['I_y'])
        Iz = float(elem['I_z'])
        J = float(elem['J'])
        G = E / (2.0 * (1.0 + nu))
        I_rho = Iy + Iz
        k_loc = _local_linear_stiffness(E, G, A, Iy, Iz, J, L)
        dofs = np.array([6 * i + 0, 6 * i + 1, 6 * i + 2, 6 * i + 3, 6 * i + 4, 6 * i + 5, 6 * j + 0, 6 * j + 1, 6 * j + 2, 6 * j + 3, 6 * j + 4, 6 * j + 5], dtype=int)
        u_e_g = u_global[dofs]
        u_e_l = T.T @ u_e_g
        f_loc = k_loc @ u_e_l
        Fx2 = float(f_loc[6])
        Mx2 = float(f_loc[9])
        My1 = float(f_loc[4])
        Mz1 = float(f_loc[5])
        My2 = float(f_loc[10])
        Mz2 = float(f_loc[11])
        k_g_loc = local_geometric_stiffness_matrix_3D_beam(L, A, I_rho, Fx2, Mx2, My1, Mz1, My2, Mz2)
        k_g_glob = T @ k_g_loc @ T.T
        K[np.ix_(dofs, dofs)] += k_g_glob
    return K