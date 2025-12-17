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
    node_coords = np.asarray(node_coords, dtype=float)
    if node_coords.ndim != 2 or node_coords.shape[1] != 3:
        raise ValueError('node_coords must be of shape (n_nodes, 3).')
    n_nodes = node_coords.shape[0]
    dof_per_node = 6
    total_dof = dof_per_node * n_nodes
    u_global = np.asarray(u_global, dtype=float)
    if u_global.ndim != 1 or u_global.size != total_dof:
        raise ValueError('u_global must be a 1D array of length 6*n_nodes.')

    def element_transformation_and_length(pi, pj, local_z_ref, is_user_provided):
        eps = 1e-12
        dx = pj - pi
        L = float(np.linalg.norm(dx))
        if not np.isfinite(L) or L <= eps:
            raise ValueError('Zero-length or invalid element length.')
        ex = dx / L
        if local_z_ref is None:
            z_try = np.array([0.0, 0.0, 1.0])
            if abs(np.dot(ex, z_try)) > 1.0 - 1e-10:
                z_try = np.array([0.0, 1.0, 0.0])
            vz = z_try
            is_user = False
        else:
            vz = np.asarray(local_z_ref, dtype=float).reshape(3)
            norm_vz = float(np.linalg.norm(vz))
            if not np.isfinite(norm_vz) or abs(norm_vz - 1.0) > 1e-08:
                raise ValueError('Provided local_z must be unit length.')
            if abs(np.dot(ex, vz)) > 1.0 - 1e-10:
                raise ValueError('Provided local_z must not be parallel to the beam axis.')
            is_user = True
        vz_perp = vz - np.dot(vz, ex) * ex
        nvz = float(np.linalg.norm(vz_perp))
        if nvz <= 1e-14:
            if is_user:
                raise ValueError('Provided local_z leads to degenerate orientation.')
            else:
                if abs(ex[0]) < 0.9:
                    tmp = np.array([1.0, 0.0, 0.0])
                else:
                    tmp = np.array([0.0, 1.0, 0.0])
                vz_perp = tmp - np.dot(tmp, ex) * ex
                nvz = float(np.linalg.norm(vz_perp))
                if nvz <= 1e-14:
                    raise ValueError('Failed to construct a valid local axis system.')
        ez = vz_perp / nvz
        ey = np.cross(ez, ex)
        ney = float(np.linalg.norm(ey))
        if ney <= 1e-14:
            raise ValueError('Failed to construct right-handed local axes.')
        ey = ey / ney
        R = np.column_stack((ex, ey, ez))
        T = np.zeros((12, 12), dtype=float)
        Rt = R.T
        for b in range(4):
            T[b * 3:(b + 1) * 3, b * 3:(b + 1) * 3] = Rt
        return (T, R, L)

    def local_elastic_stiffness(E, nu, A, Iy, Iz, J, L):
        G = E / (2.0 * (1.0 + nu))
        k = np.zeros((12, 12), dtype=float)
        EA_L = E * A / L
        k[0, 0] += EA_L
        k[0, 6] -= EA_L
        k[6, 0] -= EA_L
        k[6, 6] += EA_L
        GJ_L = G * J / L
        k[3, 3] += GJ_L
        k[3, 9] -= GJ_L
        k[9, 3] -= GJ_L
        k[9, 9] += GJ_L
        EIz = E * Iz
        L2 = L * L
        L3 = L2 * L
        cz = EIz / L3
        iz = [1, 5, 7, 11]
        kz = np.array([[12.0, 6.0 * L, -12.0, 6.0 * L], [6.0 * L, 4.0 * L2, -6.0 * L, 2.0 * L2], [-12.0, -6.0 * L, 12.0, -6.0 * L], [6.0 * L, 2.0 * L2, -6.0 * L, 4.0 * L2]], dtype=float) * cz
        for a in range(4):
            ia = iz[a]
            for b in range(4):
                ib = iz[b]
                k[ia, ib] += kz[a, b]
        EIy = E * Iy
        cy = EIy / L3
        iy = [2, 4, 8, 10]
        ky = np.array([[12.0, 6.0 * L, -12.0, 6.0 * L], [6.0 * L, 4.0 * L2, -6.0 * L, 2.0 * L2], [-12.0, -6.0 * L, 12.0, -6.0 * L], [6.0 * L, 2.0 * L2, -6.0 * L, 4.0 * L2]], dtype=float) * cy
        for a in range(4):
            ia = iy[a]
            for b in range(4):
                ib = iy[b]
                k[ia, ib] += ky[a, b]
        return k

    def local_geometric_stiffness_N(N, L):
        Kg = np.zeros((12, 12), dtype=float)
        if L <= 0.0:
            return Kg
        coef = N / (30.0 * L)
        L2 = L * L
        M = np.array([[36.0, 3.0 * L, -36.0, 3.0 * L], [3.0 * L, 4.0 * L2, -3.0 * L, -1.0 * L2], [-36.0, -3.0 * L, 36.0, -3.0 * L], [3.0 * L, -1.0 * L2, -3.0 * L, 4.0 * L2]], dtype=float) * coef
        iz = [1, 5, 7, 11]
        for a in range(4):
            ia = iz[a]
            for b in range(4):
                ib = iz[b]
                Kg[ia, ib] += M[a, b]
        iy = [2, 4, 8, 10]
        for a in range(4):
            ia = iy[a]
            for b in range(4):
                ib = iy[b]
                Kg[ia, ib] += M[a, b]
        return Kg
    K_global = np.zeros((total_dof, total_dof), dtype=float)
    for e_idx, elem in enumerate(elements):
        ni = int(elem['node_i'])
        nj = int(elem['node_j'])
        if not (0 <= ni < n_nodes and 0 <= nj < n_nodes):
            raise IndexError('Element node indices out of bounds.')
        E = float(elem['E'])
        nu = float(elem['nu'])
        A = float(elem['A'])
        Iy = float(elem['I_y'])
        Iz = float(elem['I_z'])
        J = float(elem['J'])
        local_z_ref = elem.get('local_z', None)
        Pi = node_coords[ni]
        Pj = node_coords[nj]
        T, R, L = element_transformation_and_length(Pi, Pj, local_z_ref, is_user_provided='local_z' in elem and elem['local_z'] is not None)
        dofs_i = np.arange(ni * dof_per_node, (ni + 1) * dof_per_node, dtype=int)
        dofs_j = np.arange(nj * dof_per_node, (nj + 1) * dof_per_node, dtype=int)
        edofs = np.concatenate((dofs_i, dofs_j), axis=0)
        u_e_global = u_global[edofs]
        u_local = T @ u_e_global
        k_local_elastic = local_elastic_stiffness(E, nu, A, Iy, Iz, J, L)
        f_local = k_local_elastic @ u_local
        N = 0.5 * (f_local[0] - f_local[6])
        k_g_local = local_geometric_stiffness_N(N, L)
        k_g_global_e = T.T @ k_g_local @ T
        for a_local, Aglob in enumerate(edofs):
            Ka = k_g_global_e[a_local]
            K_global[Aglob, edofs] += Ka
    K_global = 0.5 * (K_global + K_global.T)
    return K_global