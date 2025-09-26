def elastic_critical_load_analysis_frame_3D_self_contained(node_coords: np.ndarray, elements: Sequence[dict], boundary_conditions: dict[int, Sequence[int | bool]], nodal_loads: dict[int, Sequence[float]]):
    """
    Perform linear (eigenvalue) buckling analysis for a 3D frame and return the
    elastic critical load factor and associated global buckling mode shape.
    Overview
    --------
    The routine:
      1) Assembles the global elastic stiffness matrix `K`.
      2) Assembles the global reference load vector `P`.
      3) Solves the linear static problem `K u = P` (with boundary conditions) to
         obtain the displacement state `u` under the reference load.
      4) Assembles the geometric stiffness `K_g` consistent with that state.
      5) Solves the generalized eigenproblem on the free DOFs,
             K φ = -λ K_g φ,
         and returns the smallest positive eigenvalue `λ` as the elastic
         critical load factor and its corresponding global mode shape `φ`
         (constrained DOFs set to zero).
    Parameters
    ----------
    node_coords : (n_nodes, 3) ndarray of float
        Cartesian coordinates (x, y, z) for each node, indexed 0..n_nodes-1.
    elements : Sequence[dict]
        Element definitions consumed by the assembly routines. Each dictionary
        must supply properties for a 2-node 3D Euler-Bernoulli beam aligned with
        its local x-axis. Required keys (minimum):
          Topology
          --------
                Start node index (0-based).
                End node index (0-based).
          Material
          --------
                Young's modulus (used in axial, bending, and torsion terms).
                Poisson's ratio (used in torsion only, per your stiffness routine).
          Section (local axes y,z about the beam's local x)
          -----------------------------------------------
                Cross-sectional area.
                Second moment of area about local y.
                Second moment of area about local z.
                Torsional constant (for elastic/torsional stiffness).
                Polar moment about the local x-axis used by the geometric stiffness
                with torsion-bending coupling (see your geometric K routine).
          Orientation
          -----------
                Provide a 3-vector giving the direction of the element's local z-axis to 
                disambiguate the local triad used in the 12x12 transformation; if set to `None`, 
                a default convention will be applied to construct the local axes.
    boundary_conditions : dict
        Dictionary mapping node index -> boundary condition specification. Each
        node’s specification can be provided in either of two forms:
          the DOF is constrained (fixed).
          at that node are fixed.
        All constrained DOFs are removed from the free set. It is the caller’s
        responsibility to supply constraints sufficient to eliminate rigid-body
        modes.
    nodal_loads : dict[int, Sequence[float]]
        Mapping from node index → length-6 vector of load components applied at
        that node in the **global** DOF order `[F_x, F_y, F_z, M_x, M_y, M_z]`.
        Consumed by `assemble_global_load_vector_linear_elastic_3D` to form `P`.
    Returns
    -------
    elastic_critical_load_factor : float
        The smallest positive eigenvalue `λ` (> 0). If `P` is the reference load
        used to form `K_g`, then the predicted elastic buckling load is
        `P_cr = λ · P`.
    deformed_shape_vector : (6*n_nodes,) ndarray of float
        Global buckling mode vector with constrained DOFs set to zero. No
        normalization is applied (mode scale is arbitrary; only the shape matters).
    Assumptions
    -----------
      `[u_x, u_y, u_z, θ_x, θ_y, θ_z]`.
      represented via `K_g` assembled at the reference load state, not via a full
      nonlinear equilibrium/path-following analysis.
    Raises
    ------
    ValueError
        Propagated from called routines if:
    Notes
    -----
      if desired (e.g., by max absolute translational DOF).
      the returned mode can depend on numerical details.
    """

    def _get_transformation_matrix_3D(node_i_coords, node_j_coords, local_z_vec):
        v = node_j_coords - node_i_coords
        L = np.linalg.norm(v)
        if L < 1e-09:
            raise ValueError('Element length is near zero.')
        x_local = v / L
        if local_z_vec is not None:
            z_vec = np.array(local_z_vec, dtype=float)
            if np.linalg.norm(z_vec) < 1e-09:
                raise ValueError('local_z vector cannot be a zero vector.')
            z_vec /= np.linalg.norm(z_vec)
            if np.allclose(np.abs(np.dot(x_local, z_vec)), 1.0):
                raise ValueError('local_z vector cannot be collinear with the element axis.')
            y_local = np.cross(z_vec, x_local)
            y_local /= np.linalg.norm(y_local)
            z_local = np.cross(x_local, y_local)
        else:
            global_Z = np.array([0.0, 0.0, 1.0])
            if np.abs(np.abs(x_local[2]) - 1.0) > 1e-09:
                y_local = np.cross(global_Z, x_local)
                y_local /= np.linalg.norm(y_local)
                z_local = np.cross(x_local, y_local)
            else:
                global_Y = np.array([0.0, 1.0, 0.0])
                z_local = np.cross(x_local, global_Y)
                z_local /= np.linalg.norm(z_local)
                y_local = np.cross(z_local, x_local)
        R = np.vstack([x_local, y_local, z_local])
        T = np.zeros((12, 12))
        for i in range(4):
            T[i * 3:(i + 1) * 3, i * 3:(i + 1) * 3] = R
        return (T, L)

    def _get_local_elastic_stiffness_matrix_3D(L, E, G, A, Iy, Iz, J):
        k = np.zeros((12, 12))
        k[0, 0] = k[6, 6] = E * A / L
        k[0, 6] = k[6, 0] = -E * A / L
        k[3, 3] = k[9, 9] = G * J / L
        k[3, 9] = k[9, 3] = -G * J / L
        (c1, c2, c3, c4) = (12 * E * Iz / L ** 3, 6 * E * Iz / L ** 2, 4 * E * Iz / L, 2 * E * Iz / L)
        k[1, 1] = c1
        k[1, 5] = c2
        k[1, 7] = -c1
        k[1, 11] = c2
        k[5, 1] = c2
        k[5, 5] = c3
        k[5, 7] = -c2
        k[5, 11] = c4
        k[7, 1] = -c1
        k[7, 5] = -c2
        k[7, 7] = c1
        k[7, 11] = -c2
        k[11, 1] = c2
        k[11, 5] = c4
        k[11, 7] = -c2
        k[11, 11] = c3
        (c5, c6, c7, c8) = (12 * E * Iy / L ** 3, 6 * E * Iy / L ** 2, 4 * E * Iy / L, 2 * E * Iy / L)
        k[2, 2] = c5
        k[2, 4] = -c6
        k[2, 8] = -c5
        k[2, 10] = -c6
        k[4, 2] = -c6
        k[4, 4] = c7
        k[4, 8] = c6
        k[4, 10] = c8
        k[8, 2] = -c5
        k[8, 4] = c6
        k[8, 8] = c5
        k[8, 10] = c6
        k[10, 2] = -c6
        k[10, 4] = c8
        k[10, 8] = c6
        k[10, 10] = c7
        return k

    def _get_local_geometric_stiffness_matrix_3D(P_axial, L, A, I_rho):
        kg = np.zeros((12, 12))
        c1 = P_axial / (30 * L)
        kg_sub_bend = c1 * np.array([[36, 3 * L, -36, 3 * L], [3 * L, 4 * L ** 2, -3 * L, -L ** 2], [-36, -3 * L, 36, -3 * L], [3 * L, -L ** 2, -3 * L, 4 * L ** 2]])
        idx_yz = np.ix_([1, 5, 7, 11], [1, 5, 7, 11])
        kg[idx_yz] += kg_sub_bend
        idx_zy = np.ix_([2, 4, 8, 10], [2, 4, 8, 10])
        kg[idx_zy] += kg_sub_bend
        kg[2, 4] *= -1
        kg[4, 2] *= -1
        kg[2, 10] *= -1
        kg[10, 2] *= -1
        kg[8, 4] *= -1
        kg[4, 8] *= -1
        kg[8, 10] *= -1
        kg[10, 8] *= -1
        if A > 1e-12:
            c2 = P_axial * I_rho / (A * L)
            kg[3, 3] += c2
            kg[9, 9] += c2
            kg[3, 9] -= c2
            kg[9, 3] -= c2
        return kg
    n_nodes = node_coords.shape[0]
    n_dofs = 6 * n_nodes
    K = np.zeros((n_dofs, n_dofs))
    P = np.zeros(n_dofs)
    element_cache = []
    for elem in elements:
        (i, j) = (elem['node_i'], elem['node_j'])
        (coord_i, coord_j) = (node_coords[i], node_coords[j])
        (T, L) = _get_transformation_matrix_3D(coord_i, coord_j, elem.get('local_z'))
        (E, nu) = (elem['E'], elem['nu'])
        G = E / (2 * (1 + nu))
        (A, Iy, Iz, J) = (elem['A'], elem['Iy'], elem['Iz'], elem['J'])
        k_local = _get_local_elastic_stiffness_matrix_3D(L, E, G, A, Iy, Iz, J)
        k_global = T.T @ k_local @ T
        dof_indices = np.r_[6 * i:6 * i + 6, 6 * j:6 * j + 6]
        K[np.ix_(dof_indices, dof_indices)] += k_global
        element_cache.append({'T': T, 'L': L, 'dof_indices': dof_indices})
    for (node_idx, load_vec) in nodal_loads.items():
        if len(load_vec) != 6:
            raise ValueError(f'Load vector for node {node_idx} must have 6 components.')
        start_dof = 6 * node_idx
        P[start_dof:start_dof + 6] += load_vec
    constrained_dofs = np.zeros(n_dofs, dtype=bool)
    for (node_idx, bc_spec) in boundary_conditions.items():
        start_dof = 6 * node_idx
        if all((isinstance(x, bool) for x in bc_spec)):
            if len(bc_spec) != 6:
                raise ValueError(f'Boolean BC spec for node {node_idx} must have length 6.')
            constrained_dofs[start_dof:start_dof + 6] = bc_spec
        else:
            for dof_idx in bc_spec:
                if 0 <= dof_idx < 6:
                    constrained_dofs[start_dof + dof_idx] = True
                else:
                    raise ValueError(f'Invalid DOF index {dof_idx} for node {node_idx}.')
    free_dofs = np.where(~constrained_dofs)[0]
    if free_dofs.size == 0:
        raise ValueError('All DOFs are constrained; no analysis is possible.')
    K_ff = K[np.ix_(free_dofs, free_dofs)]
    P_f = P[free_dofs]
    try:
        u_f = np.linalg.solve(K_ff, P_f)
    except np.linalg.LinAlgError:
        raise ValueError('The stiffness matrix is singular. Check boundary conditions for rigid body modes.')
    u = np.zeros(n_dofs)
    u[free_dofs] = u_f
    K_g = np.zeros((n_dofs, n_dofs))
    for (i, elem) in enumerate(elements):
        cache = element_cache[i]
        (T, L, dof_indices) = (cache['T'], cache['L'], cache['dof_indices'])
        u_global_elem = u[dof_indices]
        u_local_elem = T @ u_global_elem
        P_axial = elem['E'] * elem['A'] / L * (u_local_elem[6] - u_local_elem[0])
        k_g_local = _get_local_geometric_stiffness_matrix_3D(P_axial, L, elem['A'], elem['I_rho'])
        k_g_global = T.T @ k_g_local @ T
        K_g[np.ix_(dof_indices, dof_indices)] += k_g_global
    K_g_ff = K_g[np.ix_(free_dofs, free_dofs)]
    try:
        (eigvals, eigvecs) = scipy.linalg.eigh(K_ff, -K_g_ff)
    except (np.linalg.LinAlgError, scipy.linalg.LinAlgError):
        raise ValueError('Eigenvalue analysis failed. Check model for instabilities or errors.')
    positive_eigvals_mask = eigvals > 1e-09
    if not np.any(positive_eigvals_mask):
        raise ValueError('No positive eigenvalues found, indicating no buckling load or an unstable reference state.')
    positive_eigvals = eigvals[positive_eigvals_mask]
    positive_eigvecs = eigvecs[:, positive_eigvals_mask]
    min_idx = np.argmin(positive_eigvals)
    elastic_critical_load_factor = positive_eigvals[min_idx]
    phi_f = positive_eigvecs[:, min_idx]
    deformed_shape_vector = np.zeros(n_dofs)
    deformed_shape_vector[free_dofs] = phi_f
    return (float(elastic_critical_load_factor), deformed_shape_vector)