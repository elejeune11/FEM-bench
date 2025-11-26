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
    n_nodes = node_coords.shape[0]
    n_dof = n_nodes * 6
    Kg = np.zeros((n_dof, n_dof))
    for el in elements:
        if 'connectivity' in el:
            nodes = el['connectivity']
        elif 'nodes' in el:
            nodes = el['nodes']
        else:
            raise KeyError("Element dictionary must contain 'connectivity' or 'nodes'")
        (n1, n2) = (nodes[0], nodes[1])
        p1 = node_coords[n1]
        p2 = node_coords[n2]
        delta = p2 - p1
        L = np.linalg.norm(delta)
        if L <= 0:
            raise ValueError(f'Element between nodes {n1} and {n2} has zero length.')
        x_loc = delta / L
        if 'local_z' in el and el['local_z'] is not None:
            z_ref = np.array(el['local_z'], dtype=float)
            z_ref_norm = np.linalg.norm(z_ref)
            if z_ref_norm == 0:
                raise ValueError('Provided local_z is zero vector.')
            z_ref = z_ref / z_ref_norm
        elif abs(x_loc[2]) > 0.999999:
            z_ref = np.array([0.0, 1.0, 0.0])
        else:
            z_ref = np.array([0.0, 0.0, 1.0])
        y_loc = np.cross(z_ref, x_loc)
        y_loc_norm = np.linalg.norm(y_loc)
        if y_loc_norm < 1e-12:
            raise ValueError(f'Element {n1}-{n2}: Reference vector is parallel to beam axis.')
        y_loc = y_loc / y_loc_norm
        z_loc = np.cross(x_loc, y_loc)
        R = np.vstack([x_loc, y_loc, z_loc])
        Gamma = np.zeros((12, 12))
        Gamma[0:3, 0:3] = R
        Gamma[3:6, 3:6] = R
        Gamma[6:9, 6:9] = R
        Gamma[9:12, 9:12] = R
        dof_indices = np.r_[n1 * 6:n1 * 6 + 6, n2 * 6:n2 * 6 + 6]
        u_elem_global = u_global[dof_indices]
        u_elem_local = Gamma @ u_elem_global
        E = el['E']
        nu = el['nu']
        A = el['A']
        Iy = el['I_y']
        Iz = el['I_z']
        J = el['J']
        G = E / (2.0 * (1.0 + nu))
        ke = np.zeros((12, 12))
        X = E * A / L
        T_stiff = G * J / L
        Y1 = 12 * E * Iz / L ** 3
        Y2 = 6 * E * Iz / L ** 2
        Y3 = 4 * E * Iz / L
        Y4 = 2 * E * Iz / L
        Z1 = 12 * E * Iy / L ** 3
        Z2 = 6 * E * Iy / L ** 2
        Z3 = 4 * E * Iy / L
        Z4 = 2 * E * Iy / L
        ke[0, 0] = X
        ke[0, 6] = -X
        ke[6, 0] = -X
        ke[6, 6] = X
        ke[3, 3] = T_stiff
        ke[3, 9] = -T_stiff
        ke[9, 3] = -T_stiff
        ke[9, 9] = T_stiff
        ke[1, 1] = Y1
        ke[1, 5] = Y2
        ke[1, 7] = -Y1
        ke[1, 11] = Y2
        ke[5, 1] = Y2
        ke[5, 5] = Y3
        ke[5, 7] = -Y2
        ke[5, 11] = Y4
        ke[7, 1] = -Y1
        ke[7, 5] = -Y2
        ke[7, 7] = Y1
        ke[7, 11] = -Y2
        ke[11, 1] = Y2
        ke[11, 5] = Y4
        ke[11, 7] = -Y2
        ke[11, 11] = Y3
        ke[2, 2] = Z1
        ke[2, 4] = -Z2
        ke[2, 8] = -Z1
        ke[2, 10] = -Z2
        ke[4, 2] = -Z2
        ke[4, 4] = Z3
        ke[4, 8] = Z2
        ke[4, 10] = Z4
        ke[8, 2] = -Z1
        ke[8, 4] = Z2
        ke[8, 8] = Z1
        ke[8, 10] = Z2
        ke[10, 2] = -Z2
        ke[10, 4] = Z4
        ke[10, 8] = Z2
        ke[10, 10] = Z3
        f_local = ke @ u_elem_local
        Fx2 = f_local[6]
        Mx2 = f_local[9]
        My1 = f_local[4]
        Mz1 = f_local[5]
        My2 = f_local[10]
        Mz2 = f_local[11]
        I_rho = Iy + Iz
        kg_local = local_geometric_stiffness_matrix_3D_beam(L, A, I_rho, Fx2, Mx2, My1, Mz1, My2, Mz2)
        Kg_elem_global = Gamma.T @ kg_local @ Gamma
        Kg[np.ix_(dof_indices, dof_indices)] += Kg_elem_global
    return Kg