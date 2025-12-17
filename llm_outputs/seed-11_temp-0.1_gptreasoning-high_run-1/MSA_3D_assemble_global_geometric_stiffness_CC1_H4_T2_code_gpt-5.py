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
    node_coords = np.asarray(node_coords, dtype=float)
    if node_coords.ndim != 2 or node_coords.shape[1] != 3:
        raise ValueError('node_coords must be an array of shape (n_nodes, 3).')
    n_nodes = node_coords.shape[0]
    ndof = 6 * n_nodes
    u_global = np.asarray(u_global, dtype=float).reshape(-1)
    if u_global.size != ndof:
        raise ValueError('u_global must have length 6 * n_nodes.')
    K = np.zeros((ndof, ndof), dtype=float)
    tol_len = 1e-12
    tol_parallel = 1.0 - 1e-08
    for e in elements:
        i = int(e['node_i'])
        j = int(e['node_j'])
        if not (0 <= i < n_nodes and 0 <= j < n_nodes):
            raise IndexError('Element node index out of bounds.')
        xi = node_coords[i]
        xj = node_coords[j]
        dx = xj - xi
        L = float(np.linalg.norm(dx))
        if not np.isfinite(L) or L <= tol_len:
            raise ValueError('Element has zero or invalid length.')
        ex = dx / L
        z_ref = e.get('local_z', None)
        if z_ref is None:
            z_ref = np.array([0.0, 0.0, 1.0], dtype=float)
            if abs(np.dot(ex, z_ref)) > tol_parallel:
                z_ref = np.array([0.0, 1.0, 0.0], dtype=float)
        else:
            z_ref = np.asarray(z_ref, dtype=float).reshape(3)
            nz = float(np.linalg.norm(z_ref))
            if not np.isfinite(nz) or nz <= tol_len:
                raise ValueError('Provided local_z has zero or invalid norm.')
            if abs(nz - 1.0) > 1e-06:
                raise ValueError('Provided local_z must be unit length.')
            if abs(np.dot(ex, z_ref)) > tol_parallel:
                raise ValueError('Provided local_z must not be parallel to the beam axis.')
        ey_temp = np.cross(z_ref, ex)
        ney = float(np.linalg.norm(ey_temp))
        if ney <= tol_len:
            raise ValueError('Failed to construct local element frame (degenerate local_z).')
        ey = ey_temp / ney
        ez = np.cross(ex, ey)
        ez /= np.linalg.norm(ez)
        R = np.column_stack((ex, ey, ez))
        Gamma = np.zeros((12, 12), dtype=float)
        Gamma[0:3, 0:3] = R
        Gamma[3:6, 3:6] = R
        Gamma[6:9, 6:9] = R
        Gamma[9:12, 9:12] = R
        dof_i = np.arange(6 * i, 6 * i + 6, dtype=int)
        dof_j = np.arange(6 * j, 6 * j + 6, dtype=int)
        edofs = np.hstack((dof_i, dof_j))
        u_e_global = u_global[edofs]
        u_local = Gamma.T @ u_e_global
        E = float(e['E'])
        nu = float(e['nu'])
        A = float(e['A'])
        I_y = float(e['I_y'])
        I_z = float(e['I_z'])
        J = float(e['J'])
        G = E / (2.0 * (1.0 + nu))
        Ke = np.zeros((12, 12), dtype=float)
        EA_L = E * A / L
        Ke[0, 0] += EA_L
        Ke[0, 6] -= EA_L
        Ke[6, 0] -= EA_L
        Ke[6, 6] += EA_L
        GJ_L = G * J / L
        Ke[3, 3] += GJ_L
        Ke[3, 9] -= GJ_L
        Ke[9, 3] -= GJ_L
        Ke[9, 9] += GJ_L
        EIz = E * I_z
        k1 = 12.0 * EIz / L ** 3
        k2 = 6.0 * EIz / L ** 2
        k3 = 4.0 * EIz / L
        k4 = 2.0 * EIz / L
        a, b, c, d = (1, 5, 7, 11)
        Ke[a, a] += k1
        Ke[a, b] += k2
        Ke[a, c] += -k1
        Ke[a, d] += k2
        Ke[b, a] += k2
        Ke[b, b] += k3
        Ke[b, c] += -k2
        Ke[b, d] += k4
        Ke[c, a] += -k1
        Ke[c, b] += -k2
        Ke[c, c] += k1
        Ke[c, d] += -k2
        Ke[d, a] += k2
        Ke[d, b] += k4
        Ke[d, c] += -k2
        Ke[d, d] += k3
        EIy = E * I_y
        k1y = 12.0 * EIy / L ** 3
        k2y = 6.0 * EIy / L ** 2
        k3y = 4.0 * EIy / L
        k4y = 2.0 * EIy / L
        a, b, c, d = (2, 4, 8, 10)
        Ke[a, a] += k1y
        Ke[a, b] += -k2y
        Ke[a, c] += -k1y
        Ke[a, d] += -k2y
        Ke[b, a] += -k2y
        Ke[b, b] += k3y
        Ke[b, c] += k2y
        Ke[b, d] += k4y
        Ke[c, a] += -k1y
        Ke[c, b] += k2y
        Ke[c, c] += k1y
        Ke[c, d] += k2y
        Ke[d, a] += -k2y
        Ke[d, b] += k4y
        Ke[d, c] += k2y
        Ke[d, d] += k3y
        f_local = Ke @ u_local
        Fx2 = float(f_local[6])
        Mx2 = float(f_local[9])
        My1 = float(f_local[4])
        Mz1 = float(f_local[5])
        My2 = float(f_local[10])
        Mz2 = float(f_local[11])
        I_rho = I_y + I_z
        k_g_local = local_geometric_stiffness_matrix_3D_beam(L=L, A=A, I_rho=I_rho, Fx2=Fx2, Mx2=Mx2, My1=My1, Mz1=Mz1, My2=My2, Mz2=Mz2)
        k_g_global = Gamma @ k_g_local @ Gamma.T
        K[np.ix_(edofs, edofs)] += k_g_global
    return K