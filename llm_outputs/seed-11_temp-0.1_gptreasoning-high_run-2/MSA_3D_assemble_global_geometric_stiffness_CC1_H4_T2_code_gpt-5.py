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
    External Dependencies
    ---------------------
    local_geometric_stiffness_matrix_3D_beam(L, A, I_rho, Fx2, Mx2, My1, Mz1, My2, Mz2) -> (12,12) ndarray
        Must return the local geometric stiffness using the element length L, section properties, and local end force resultants as shown.
    """
    n_nodes = node_coords.shape[0]
    dof_per_node = 6
    n_dofs = dof_per_node * n_nodes
    K = np.zeros((n_dofs, n_dofs), dtype=float)
    tol_len = 1e-12
    tol_parallel = 1e-10
    tol_unit = 1e-08

    def compute_rotation_and_length(pi: np.ndarray, pj: np.ndarray, local_z_opt) -> tuple:
        dx = pj - pi
        L = np.linalg.norm(dx)
        if not np.isfinite(L) or L <= tol_len:
            raise ValueError('Zero or invalid element length.')
        ex = dx / L
        if local_z_opt is None:
            z_ref = np.array([0.0, 0.0, 1.0], dtype=float)
            if abs(np.dot(ex, z_ref)) > 1.0 - tol_parallel:
                z_ref = np.array([0.0, 1.0, 0.0], dtype=float)
        else:
            z_ref = np.asarray(local_z_opt, dtype=float)
            nrm = np.linalg.norm(z_ref)
            if not np.isfinite(nrm) or abs(nrm - 1.0) > tol_unit:
                raise ValueError('Provided local_z must be unit length.')
            if abs(np.dot(ex, z_ref)) > 1.0 - tol_parallel:
                raise ValueError('Provided local_z is parallel to element axis.')
        ey_temp = np.cross(z_ref, ex)
        n_ey = np.linalg.norm(ey_temp)
        if n_ey <= tol_parallel:
            raise ValueError('Invalid local_z; nearly parallel to element axis.')
        ey = ey_temp / n_ey
        ez = np.cross(ex, ey)
        nez = np.linalg.norm(ez)
        if nez <= tol_parallel:
            raise ValueError('Failed to construct orthonormal triad.')
        ez = ez / nez
        R = np.vstack((ex, ey, ez))
        return (R, L)

    def local_elastic_stiffness(E: float, G: float, A: float, Iy: float, Iz: float, J: float, L: float) -> np.ndarray:
        k = np.zeros((12, 12), dtype=float)
        EA_L = E * A / L
        k[0, 0] = EA_L
        k[0, 6] = -EA_L
        k[6, 0] = -EA_L
        k[6, 6] = EA_L
        GJ_L = G * J / L
        k[3, 3] = GJ_L
        k[3, 9] = -GJ_L
        k[9, 3] = -GJ_L
        k[9, 9] = GJ_L
        EIz = E * Iz
        fz1 = 12.0 * EIz / L ** 3
        fz2 = 6.0 * EIz / L ** 2
        fz3 = 4.0 * EIz / L
        fz4 = 2.0 * EIz / L
        k[1, 1] = fz1
        k[1, 5] = fz2
        k[1, 7] = -fz1
        k[1, 11] = fz2
        k[5, 1] = fz2
        k[5, 5] = fz3
        k[5, 7] = -fz2
        k[5, 11] = fz4
        k[7, 1] = -fz1
        k[7, 5] = -fz2
        k[7, 7] = fz1
        k[7, 11] = -fz2
        k[11, 1] = fz2
        k[11, 5] = fz4
        k[11, 7] = -fz2
        k[11, 11] = fz3
        EIy = E * Iy
        fy1 = 12.0 * EIy / L ** 3
        fy2 = 6.0 * EIy / L ** 2
        fy3 = 4.0 * EIy / L
        fy4 = 2.0 * EIy / L
        k[2, 2] = fy1
        k[2, 4] = fy2
        k[2, 8] = -fy1
        k[2, 10] = fy2
        k[4, 2] = fy2
        k[4, 4] = fy3
        k[4, 8] = -fy2
        k[4, 10] = fy4
        k[8, 2] = -fy1
        k[8, 4] = -fy2
        k[8, 8] = fy1
        k[8, 10] = -fy2
        k[10, 2] = fy2
        k[10, 4] = fy4
        k[10, 8] = -fy2
        k[10, 10] = fy3
        return k
    for elem in elements:
        i = int(elem['node_i'])
        j = int(elem['node_j'])
        pi = np.asarray(node_coords[i], dtype=float)
        pj = np.asarray(node_coords[j], dtype=float)
        R, L = compute_rotation_and_length(pi, pj, elem.get('local_z', None))
        E = float(elem['E'])
        nu = float(elem['nu'])
        A = float(elem['A'])
        Iy = float(elem['I_y'])
        Iz = float(elem['I_z'])
        J = float(elem['J'])
        G = E / (2.0 * (1.0 + nu))
        I_rho = Iy + Iz
        edofs = np.r_[np.arange(6 * i, 6 * i + 6), np.arange(6 * j, 6 * j + 6)]
        u_e_global = u_global[edofs]
        d_local = np.zeros(12, dtype=float)
        d_local[0:3] = R @ u_e_global[0:3]
        d_local[3:6] = R @ u_e_global[3:6]
        d_local[6:9] = R @ u_e_global[6:9]
        d_local[9:12] = R @ u_e_global[9:12]
        k_e_local = local_elastic_stiffness(E, G, A, Iy, Iz, J, L)
        f_local = k_e_local @ d_local
        Fx2 = float(f_local[6])
        Mx2 = float(f_local[9])
        My1 = float(f_local[4])
        Mz1 = float(f_local[5])
        My2 = float(f_local[10])
        Mz2 = float(f_local[11])
        k_g_local = local_geometric_stiffness_matrix_3D_beam(L=L, A=A, I_rho=I_rho, Fx2=Fx2, Mx2=Mx2, My1=My1, Mz1=Mz1, My2=My2, Mz2=Mz2)
        Gamma = np.zeros((12, 12), dtype=float)
        Gamma[0:3, 0:3] = R
        Gamma[3:6, 3:6] = R
        Gamma[6:9, 6:9] = R
        Gamma[9:12, 9:12] = R
        k_g_global_e = Gamma.T @ k_g_local @ Gamma
        K[np.ix_(edofs, edofs)] += k_g_global_e
    return K