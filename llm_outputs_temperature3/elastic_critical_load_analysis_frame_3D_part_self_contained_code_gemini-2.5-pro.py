def elastic_critical_load_analysis_frame_3D_part_self_contained(node_coords: np.ndarray, elements: Sequence[dict], boundary_conditions: dict[int, Sequence[int | bool]], nodal_loads: dict[int, Sequence[float]]):
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
                Poisson's ratio (used in torsion only).
          Section (local axes y,z about the beam's local x)
          -----------------------------------------------
                Cross-sectional area.
                Second moment of area about local y.
                Second moment of area about local z.
                Torsional constant (for elastic/torsional stiffness).
                Polar moment about the local x-axis used by the geometric stiffness
                with torsion–bending coupling.
          Orientation
          -----------
                Provide a 3-vector giving the direction of the element's local z-axis to 
                disambiguate the local triad used in the 12×12 transformation; if `None`, 
                a default convention is applied.
    boundary_conditions : dict
        Dictionary mapping node index -> boundary condition specification. Each
        node's specification can be provided in either of two forms:
          the DOF is constrained (fixed).
          at that node are fixed.
        All constrained DOFs are removed from the free set. It is the caller’s
        responsibility to supply constraints sufficient to eliminate rigid-body
        modes.
    nodal_loads : dict[int, Sequence[float]]
        Mapping from node index → length-6 vector of load components applied at
        that node in the **global** DOF order `[F_x, F_y, F_z, M_x, M_y, M_z]`.
        Used to form `P`.
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
    External Helper Functions (required)
    ------------------------------------
        Local elastic stiffness matrix for a 3D Euler-Bernoulli beam aligned with
        the local x-axis.
        Local geometric stiffness matrix with torsion-bending coupling.
    Raises
    ------
    ValueError
        Propagated from called routines if:
    Notes
    -----
      if desired (e.g., by max absolute translational DOF).
      the returned mode can depend on numerical details.
    """
    n_nodes = node_coords.shape[0]
    n_dof = n_nodes * 6
    K = np.zeros((n_dof, n_dof))
    P = np.zeros(n_dof)
    constrained_dofs = np.zeros(n_dof, dtype=bool)
    for (node_idx, bc_spec) in boundary_conditions.items():
        start_dof = node_idx * 6
        if all((isinstance(x, bool) for x in bc_spec)):
            for (i, is_constrained) in enumerate(bc_spec):
                if is_constrained:
                    constrained_dofs[start_dof + i] = True
        else:
            for dof_local_idx in bc_spec:
                constrained_dofs[start_dof + dof_local_idx] = True
    free_dofs = np.where(~constrained_dofs)[0]
    for (node_idx, loads) in nodal_loads.items():
        start_idx = node_idx * 6
        P[start_idx:start_idx + 6] += loads
    element_data_cache = []
    for elem in elements:
        (i, j) = (elem['node_i'], elem['node_j'])
        (p1, p2) = (node_coords[i], node_coords[j])
        vec_ij = p2 - p1
        L = np.linalg.norm(vec_ij)
        if L < 1e-09:
            raise ValueError(f'Element between nodes {i} and {j} has zero length.')
        x_local = vec_ij / L
        if elem.get('local_z') is not None:
            v_orient = np.array(elem['local_z'], dtype=float)
        else:
            global_z = np.array([0.0, 0.0, 1.0])
            if np.allclose(np.abs(np.dot(x_local, global_z)), 1.0):
                v_orient = np.array([1.0, 0.0, 0.0])
            else:
                v_orient = global_z
        if np.linalg.norm(np.cross(v_orient, x_local)) < 1e-09:
            raise ValueError(f'Orientation vector for element {i}-{j} is parallel to the element axis.')
        y_local = np.cross(v_orient, x_local)
        y_local /= np.linalg.norm(y_local)
        z_local = np.cross(x_local, y_local)
        R_3x3 = np.array([x_local, y_local, z_local])
        T_elem = scipy.linalg.block_diag(R_3x3, R_3x3, R_3x3, R_3x3)
        k_e = local_elastic_stiffness_matrix_3D_beam(elem['E'], elem['nu'], elem['A'], L, elem['Iy'], elem['Iz'], elem['J'])
        K_e_global = T_elem.T @ k_e @ T_elem
        dof_map = np.concatenate((np.arange(i * 6, i * 6 + 6), np.arange(j * 6, j * 6 + 6)))
        K[np.ix_(dof_map, dof_map)] += K_e_global
        element_data_cache.append({'L': L, 'T_elem': T_elem, 'k_e': k_e, 'dof_map': dof_map})
    if len(free_dofs) == 0:
        raise ValueError('No free degrees of freedom. The structure is fully constrained.')
    K_free = K[np.ix_(free_dofs, free_dofs)]
    P_free = P[free_dofs]
    try:
        u_free = scipy.linalg.solve(K_free, P_free, assume_a='sym')
    except np.linalg.LinAlgError:
        raise ValueError('The stiffness matrix is singular. Check boundary conditions for rigid body modes.')
    u = np.zeros(n_dof)
    u[free_dofs] = u_free
    K_g = np.zeros((n_dof, n_dof))
    for (idx, elem) in enumerate(elements):
        cached_data = element_data_cache[idx]
        (L, T_elem, k_e, dof_map) = (cached_data['L'], cached_data['T_elem'], cached_data['k_e'], cached_data['dof_map'])
        u_e_global = u[dof_map]
        u_e_local = T_elem @ u_e_global
        f_e_local = k_e @ u_e_local
        Fx2 = f_e_local[6]
        Mx2 = f_e_local[9]
        My1 = f_e_local[4]
        Mz1 = f_e_local[5]
        My2 = f_e_local[10]
        Mz2 = f_e_local[11]
        k_g = local_geometric_stiffness_matrix_3D_beam(L, elem['A'], elem['I_rho'], Fx2, Mx2, My1, Mz1, My2, Mz2)
        K_g_global = T_elem.T @ k_g @ T_elem
        K_g[np.ix_(dof_map, dof_map)] += K_g_global
    K_g_free = K_g[np.ix_(free_dofs, free_dofs)]
    try:
        (mu_vals, phi_free_vecs) = scipy.linalg.eigh(K_g_free, b=K_free)
    except (np.linalg.LinAlgError, ValueError) as e:
        raise ValueError(f'Eigenvalue problem failed to solve: {e}')
    negative_mu_mask = mu_vals < -1e-09
    if not np.any(negative_mu_mask):
        raise ValueError('No buckling mode found (no negative mu). The applied load may be stabilizing.')
    negative_mu_indices = np.where(negative_mu_mask)[0]
    min_mu_idx_in_subset = np.argmin(mu_vals[negative_mu_indices])
    buckling_mode_idx = negative_mu_indices[min_mu_idx_in_subset]
    mu_crit = mu_vals[buckling_mode_idx]
    elastic_critical_load_factor = -1.0 / mu_crit
    phi_free = phi_free_vecs[:, buckling_mode_idx]
    deformed_shape_vector = np.zeros(n_dof)
    deformed_shape_vector[free_dofs] = phi_free
    return (elastic_critical_load_factor, deformed_shape_vector)