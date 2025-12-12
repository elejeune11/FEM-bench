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
    n_nodes = int(node_coords.shape[0])
    ndof = 6 * n_nodes
    if u_global.ndim != 1 or u_global.size != ndof:
        raise ValueError('u_global must be a 1D array of length 6*n_nodes.')
    K = np.zeros((ndof, ndof), dtype=float)

    def _extract_nodes(ed: dict):
        if 'nodes' in ed:
            pair = ed['nodes']
        elif 'node_ids' in ed:
            pair = ed['node_ids']
        elif 'conn' in ed:
            pair = ed['conn']
        elif 'connectivity' in ed:
            pair = ed['connectivity']
        elif 'i' in ed and 'j' in ed:
            pair = (ed['i'], ed['j'])
        elif 'node_i' in ed and 'node_j' in ed:
            pair = (ed['node_i'], ed['node_j'])
        elif 'n1' in ed and 'n2' in ed:
            pair = (ed['n1'], ed['n2'])
        else:
            raise KeyError("Element dictionary missing node connectivity. Expected keys like 'nodes', 'conn', or ('i','j').")
        if len(pair) != 2:
            raise ValueError('Element connectivity must specify exactly 2 nodes.')
        (n1, n2) = (int(pair[0]), int(pair[1]))
        if not (0 <= n1 < n_nodes and 0 <= n2 < n_nodes):
            raise IndexError('Element node indices out of range.')
        if n1 == n2:
            raise ValueError('Zero-length element: identical node indices.')
        return (n1, n2)
    tol_parallel = 1.0 - 1e-08
    tol_len = 1e-12
    for elem in elements:
        (n1, n2) = _extract_nodes(elem)
        x1 = np.asarray(node_coords[n1], dtype=float)
        x2 = np.asarray(node_coords[n2], dtype=float)
        dx = x2 - x1
        L = float(np.linalg.norm(dx))
        if not np.isfinite(L) or L <= tol_len:
            raise ValueError('Zero-length or invalid element length.')
        ex = dx / L
        ref = elem.get('local_z', None)
        if ref is None:
            ref_vec = np.array([0.0, 0.0, 1.0], dtype=float)
            if abs(float(np.dot(ex, ref_vec))) > tol_parallel:
                ref_vec = np.array([0.0, 1.0, 0.0], dtype=float)
        else:
            ref_vec = np.asarray(ref, dtype=float).reshape(3)
            nrm = float(np.linalg.norm(ref_vec))
            if not np.isfinite(nrm) or nrm <= tol_len:
                raise ValueError('Provided local_z must be a non-zero vector.')
            ref_vec = ref_vec / nrm
            if abs(float(np.dot(ref_vec, ex))) > tol_parallel:
                raise ValueError('Provided local_z is parallel to the element axis.')
        ey_temp = np.cross(ref_vec, ex)
        ny = float(np.linalg.norm(ey_temp))
        if ny <= tol_len:
            raise ValueError('Cannot construct local frame; reference vector parallel to axis.')
        ey = ey_temp / ny
        ez = np.cross(ex, ey)
        R = np.vstack((ex, ey, ez))
        T = np.zeros((12, 12), dtype=float)
        T[0:3, 0:3] = R
        T[3:6, 3:6] = R
        T[6:9, 6:9] = R
        T[9:12, 9:12] = R
        dof_idx = np.array([6 * n1 + 0, 6 * n1 + 1, 6 * n1 + 2, 6 * n1 + 3, 6 * n1 + 4, 6 * n1 + 5, 6 * n2 + 0, 6 * n2 + 1, 6 * n2 + 2, 6 * n2 + 3, 6 * n2 + 4, 6 * n2 + 5], dtype=int)
        d_g = u_global[dof_idx]
        d_l = T @ d_g
        E = float(elem['E'])
        nu = float(elem['nu'])
        A = float(elem['A'])
        Iy = float(elem['I_y'])
        Iz = float(elem['I_z'])
        J = float(elem['J'])
        G = E / (2.0 * (1.0 + nu))
        k_e = np.zeros((12, 12), dtype=float)
        a_ax = E * A / L
        k_e[0, 0] += a_ax
        k_e[0, 6] += -a_ax
        k_e[6, 0] += -a_ax
        k_e[6, 6] += a_ax
        k_t = G * J / L
        k_e[3, 3] += k_t
        k_e[3, 9] += -k_t
        k_e[9, 3] += -k_t
        k_e[9, 9] += k_t
        bz = 12.0 * E * Iz / L ** 3
        cz = 6.0 * E * Iz / L ** 2
        dz = 4.0 * E * Iz / L
        ezc = 2.0 * E * Iz / L
        k_e[1, 1] += bz
        k_e[1, 5] += cz
        k_e[1, 7] += -bz
        k_e[1, 11] += cz
        k_e[5, 1] += cz
        k_e[5, 5] += dz
        k_e[5, 7] += -cz
        k_e[5, 11] += ezc
        k_e[7, 1] += -bz
        k_e[7, 5] += -cz
        k_e[7, 7] += bz
        k_e[7, 11] += -cz
        k_e[11, 1] += cz
        k_e[11, 5] += ezc
        k_e[11, 7] += -cz
        k_e[11, 11] += dz
        by_ = 12.0 * E * Iy / L ** 3
        cy_ = 6.0 * E * Iy / L ** 2
        dy_ = 4.0 * E * Iy / L
        ey_ = 2.0 * E * Iy / L
        k_e[2, 2] += by_
        k_e[2, 4] += -cy_
        k_e[2, 8] += -by_
        k_e[2, 10] += -cy_
        k_e[4, 2] += -cy_
        k_e[4, 4] += dy_
        k_e[4, 8] += cy_
        k_e[4, 10] += ey_
        k_e[8, 2] += -by_
        k_e[8, 4] += cy_
        k_e[8, 8] += by_
        k_e[8, 10] += cy_
        k_e[10, 2] += -cy_
        k_e[10, 4] += ey_
        k_e[10, 8] += cy_
        k_e[10, 10] += dy_
        f_int_local = k_e @ d_l
        Fx2 = float(f_int_local[6])
        Mx2 = float(f_int_local[9])
        My1 = float(f_int_local[4])
        Mz1 = float(f_int_local[5])
        My2 = float(f_int_local[10])
        Mz2 = float(f_int_local[11])
        I_rho = J
        k_g_local = local_geometric_stiffness_matrix_3D_beam(L=L, A=A, I_rho=I_rho, Fx2=Fx2, Mx2=Mx2, My1=My1, Mz1=Mz1, My2=My2, Mz2=Mz2)
        k_g_global_elem = T.T @ k_g_local @ T
        K[np.ix_(dof_idx, dof_idx)] += k_g_global_elem
    return K