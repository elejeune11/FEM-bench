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
    n_nodes = node_coords.shape[0]
    K_g = np.zeros((6 * n_nodes, 6 * n_nodes))
    for el in elements:
        if 'connectivity' in el:
            conn = el['connectivity']
        elif 'nodes' in el:
            conn = el['nodes']
        else:
            raise KeyError("Element definition must contain 'connectivity' or 'nodes'")
        (i, j) = conn
        p1 = node_coords[i]
        p2 = node_coords[j]
        vec = p2 - p1
        L = np.linalg.norm(vec)
        if L < 1e-12:
            raise ValueError('Zero length element found.')
        E = el['E']
        nu = el['nu']
        A = el['A']
        Iy = el['I_y']
        Iz = el['I_z']
        J = el['J']
        G = E / (2.0 * (1.0 + nu))
        x_local = vec / L
        local_z_in = el.get('local_z', None)
        if local_z_in is not None:
            ref_vec = np.array(local_z_in, dtype=float)
            if abs(np.linalg.norm(ref_vec) - 1.0) > 1e-06:
                raise ValueError('local_z must be a unit vector.')
            y_temp = np.cross(ref_vec, x_local)
            if np.linalg.norm(y_temp) < 1e-06:
                raise ValueError('local_z cannot be parallel to the beam axis.')
            y_local = y_temp / np.linalg.norm(y_temp)
            z_local = np.cross(x_local, y_local)
        else:
            if abs(x_local[2]) > 0.999999:
                ref_vec = np.array([0.0, 1.0, 0.0])
            else:
                ref_vec = np.array([0.0, 0.0, 1.0])
            y_temp = np.cross(ref_vec, x_local)
            y_local = y_temp / np.linalg.norm(y_temp)
            z_local = np.cross(x_local, y_local)
        R = np.vstack([x_local, y_local, z_local])
        Gamma = np.zeros((12, 12))
        Gamma[0:3, 0:3] = R
        Gamma[3:6, 3:6] = R
        Gamma[6:9, 6:9] = R
        Gamma[9:12, 9:12] = R
        idx = np.r_[6 * i:6 * i + 6, 6 * j:6 * j + 6]
        u_el_glob = u_global[idx]
        u_el_loc = Gamma @ u_el_glob
        k_e = np.zeros((12, 12))
        X = E * A / L
        T = G * J / L
        Az = 12.0 * E * Iz / L ** 3
        Bz = 6.0 * E * Iz / L ** 2
        Cz = 4.0 * E * Iz / L
        Dz = 2.0 * E * Iz / L
        Ay = 12.0 * E * Iy / L ** 3
        By = 6.0 * E * Iy / L ** 2
        Cy = 4.0 * E * Iy / L
        Dy = 2.0 * E * Iy / L
        k_e[0, 0] = X
        k_e[0, 6] = -X
        k_e[6, 0] = -X
        k_e[6, 6] = X
        k_e[3, 3] = T
        k_e[3, 9] = -T
        k_e[9, 3] = -T
        k_e[9, 9] = T
        k_e[1, 1] = Az
        k_e[1, 5] = Bz
        k_e[1, 7] = -Az
        k_e[1, 11] = Bz
        k_e[5, 1] = Bz
        k_e[5, 5] = Cz
        k_e[5, 7] = -Bz
        k_e[5, 11] = Dz
        k_e[7, 1] = -Az
        k_e[7, 5] = -Bz
        k_e[7, 7] = Az
        k_e[7, 11] = -Bz
        k_e[11, 1] = Bz
        k_e[11, 5] = Dz
        k_e[11, 7] = -Bz
        k_e[11, 11] = Cz
        k_e[2, 2] = Ay
        k_e[2, 4] = -By
        k_e[2, 8] = -Ay
        k_e[2, 10] = -By
        k_e[4, 2] = -By
        k_e[4, 4] = Cy
        k_e[4, 8] = By
        k_e[4, 10] = Dy
        k_e[8, 2] = -Ay
        k_e[8, 4] = By
        k_e[8, 8] = Ay
        k_e[8, 10] = By
        k_e[10, 2] = -By
        k_e[10, 4] = Dy
        k_e[10, 8] = By
        k_e[10, 10] = Cy
        f_loc = k_e @ u_el_loc
        P = f_loc[6]
        Mx2 = f_loc[9]
        My1 = f_loc[4]
        My2 = f_loc[10]
        Mz1 = f_loc[5]
        Mz2 = f_loc[11]
        kg = np.zeros((12, 12))
        c = P / (30.0 * L)
        kg[1, 1] += 36 * c
        kg[1, 5] += 3 * L * c
        kg[1, 7] += -36 * c
        kg[1, 11] += 3 * L * c
        kg[5, 1] += 3 * L * c
        kg[5, 5] += 4 * L ** 2 * c
        kg[5, 7] += -3 * L * c
        kg[5, 11] += -L ** 2 * c
        kg[7, 1] += -36 * c
        kg[7, 5] += -3 * L * c
        kg[7, 7] += 36 * c
        kg[7, 11] += -3 * L * c
        kg[11, 1] += 3 * L * c
        kg[11, 5] += -L ** 2 * c
        kg[11, 7] += -3 * L * c
        kg[11, 11] += 4 * L ** 2 * c
        kg[2, 2] += 36 * c
        kg[2, 4] += -3 * L * c
        kg[2, 8] += -36 * c
        kg[2, 10] += -3 * L * c
        kg[4, 2] += -3 * L * c
        kg[4, 4] += 4 * L ** 2 * c
        kg[4, 8] += 3 * L * c
        kg[4, 10] += -L ** 2 * c
        kg[8, 2] += -36 * c
        kg[8, 4] += 3 * L * c
        kg[8, 8] += 36 * c
        kg[8, 10] += 3 * L * c
        kg[10, 2] += -3 * L * c
        kg[10, 4] += -L ** 2 * c
        kg[10, 8] += 3 * L * c
        kg[10, 10] += 4 * L ** 2 * c
        inv30L = 1.0 / (30.0 * L)
        t1_3 = (2 * My1 - My2) * inv30L
        t1_9 = (My1 + My2) * inv30L
        t3_7 = t1_9
        t7_9 = -(My1 - 2 * My2) * inv30L
        kg[1, 3] += t1_3
        kg[3, 1] += t1_3
        kg[1, 9] += t1_9
        kg[9, 1] += t1_9
        kg[3, 7] += t3_7
        kg[7, 3] += t3_7
        kg[7, 9] += t7_9
        kg[9, 7] += t7_9
        t2_3 = -(2 * Mz1 - Mz2) * inv30L
        t2_9 = -(Mz1 + Mz2) * inv30L
        t3_8 = t2_9
        t8_9 = (Mz1 - 2 * Mz2) * inv30L
        kg[2, 3] += t2_3
        kg[3, 2] += t2_3
        kg[2, 9] += t2_9
        kg[9, 2] += t2_9
        kg[3, 8] += t3_8
        kg[8, 3] += t3_8
        kg[8, 9] += t8_9
        kg[9, 8] += t8_9
        tm = Mx2 / 2.0
        kg[4, 5] += tm
        kg[5, 4] += tm
        kg[4, 11] += -tm
        kg[11, 4] += -tm
        kg[10, 5] += -tm
        kg[5, 10] += -tm
        kg[10, 11] += tm
        kg[11, 10] += tm
        k_glob_el = Gamma.T @ kg @ Gamma
        for r in range(12):
            g_r = idx[r]
            for c_col in range(12):
                g_c = idx[c_col]
                K_g[g_r, g_c] += k_glob_el[r, c_col]
    return K_g