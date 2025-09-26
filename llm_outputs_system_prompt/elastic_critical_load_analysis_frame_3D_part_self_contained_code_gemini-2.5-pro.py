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
    element_data_cache = []
    for el_props in elements:
        (i, j) = (el_props['node_i'], el_props['node_j'])
        (coord_i, coord_j) = (node_coords[i], node_coords[j])
        delta = coord_j - coord_i
        L = np.linalg.norm(delta)
        if np.isclose(L, 0.0):
            raise ValueError(f'Element between nodes {i} and {j} has zero length.')
        x_local = delta / L
        if el_props['local_z'] is not None:
            v_z = np.array(el_props['local_z'], dtype=float)
            v_z_norm = np.linalg.norm(v_z)
            if np.isclose(v_z_norm, 0.0):
                raise ValueError("Provided 'local_z' vector cannot be a zero vector.")
            v_z /= v_z_norm
            if np.isclose(np.abs(np.dot(x_local, v_z)), 1.0):
                raise ValueError("Element's 'local_z' vector cannot be parallel to its axis.")
            y_local_unnorm = np.cross(v_z, x_local)
            y_local = y_local_unnorm / np.linalg.norm(y_local_unnorm)
            z_local = np.cross(x_local, y_local)
        else:
            global_Z = np.array([0.0, 0.0, 1.0])
            if np.isclose(np.abs(x_local[2]), 1.0):
                global_Y = np.array([0.0, 1.0, 0.0])
                y_local_unnorm = np.cross(global_Y, x_local)
                y_local = y_local_unnorm / np.linalg.norm(y_local_unnorm)
                z_local = np.cross(x_local, y_local)
            else:
                y_local_unnorm = np.cross(global_Z, x_local)
                y_local = y_local_unnorm / np.linalg.norm(y_local_unnorm)
                z_local = np.cross(x_local, y_local)
        R = np.vstack([x_local, y_local, z_local])
        T_12 = scipy.linalg.block_diag(R, R, R, R)
        k_e = local_elastic_stiffness_matrix_3D_beam(el_props['E'], el_props['nu'], el_props['A'], L, el_props['Iy'], el_props['Iz'], el_props['J'])
        k_global = T_12.T @ k_e @ T_12
        dof_map = np.r_[6 * i:6 * i + 6, 6 * j:6 * j + 6]
        K[np.ix_(dof_map, dof_map)] += k_global
        element_data_cache.append({'dof_map': dof_map, 'T_12': T_12, 'k_e': k_e, 'L': L, 'A': el_props['A'], 'I_rho': el_props['I_rho']})
    for (node_idx, loads) in nodal_loads.items():
        P[6 * node_idx:6 * node_idx + 6] += loads
    constrained_dofs_mask = np.zeros(n_dof, dtype=bool)
    for (node_idx, bcs) in boundary_conditions.items():
        start_dof = 6 * node_idx
        if bcs and isinstance(bcs[0], (bool, np.bool_)):
            for (k, is_constrained) in enumerate(bcs):
                if is_constrained:
                    constrained_dofs_mask[start_dof + k] = True
        else:
            for dof_local_idx in bcs:
                constrained_dofs_mask[start_dof + dof_local_idx] = True
    free_dofs = np.where(~constrained_dofs_mask)[0]
    if free_dofs.size == 0:
        raise ValueError('No free degrees of freedom.')
    K_free = K[np.ix_(free_dofs, free_dofs)]
    P_free = P[free_dofs]
    try:
        u_free = np.linalg.solve(K_free, P_free)
    except np.linalg.LinAlgError:
        raise ValueError('Singular stiffness matrix after applying BCs. Check for rigid-body modes.')
    u = np.zeros(n_dof)
    u[free_dofs] = u_free
    K_g = np.zeros((n_dof, n_dof))
    for el_data in element_data_cache:
        dof_map = el_data['dof_map']
        T_12 = el_data['T_12']
        k_e = el_data['k_e']
        u_e_global = u[dof_map]
        u_e_local = T_12 @ u_e_global
        f_local = k_e @ u_e_local
        My1 = f_local[4]
        Mz1 = f_local[5]
        Fx2 = f_local[6]
        Mx2 = f_local[9]
        My2 = f_local[10]
        Mz2 = f_local[11]
        k_g_local = local_geometric_stiffness_matrix_3D_beam(el_data['L'], el_data['A'], el_data['I_rho'], Fx2, Mx2, My1, Mz1, My2, Mz2)
        k_g_global = T_12.T @ k_g_local @ T_12
        K_g[np.ix_(dof_map, dof_map)] += k_g_global
    K_g_free = K_g[np.ix_(free_dofs, free_dofs)]
    try:
        (eigenvalues, eigenvectors) = scipy.linalg.eigh(K_free, -K_g_free)
    except (np.linalg.LinAlgError, scipy.linalg.LinAlgError) as e:
        raise ValueError(f'Eigenvalue problem failed to solve: {e}')
    positive_mask = eigenvalues > 1e-09
    if not np.any(positive_mask):
        raise ValueError('No positive eigenvalue found, indicating no buckling load or an unstable structure under the reference load.')
    positive_eigenvalues = eigenvalues[positive_mask]
    min_lambda = np.min(positive_eigenvalues)
    min_idx = np.where(eigenvalues == min_lambda)[0][0]
    phi_free = eigenvectors[:, min_idx]
    phi = np.zeros(n_dof)
    phi[free_dofs] = phi_free
    return (min_lambda, phi)