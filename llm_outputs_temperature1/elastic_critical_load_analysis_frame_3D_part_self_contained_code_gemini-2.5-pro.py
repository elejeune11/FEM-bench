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
    element_data = []
    for el in elements:
        node_i_idx = el['node_i']
        node_j_idx = el['node_j']
        coord_i = node_coords[node_i_idx]
        coord_j = node_coords[node_j_idx]
        vec_ij = coord_j - coord_i
        L = np.linalg.norm(vec_ij)
        if L < 1e-09:
            raise ValueError(f'Element between nodes {node_i_idx} and {node_j_idx} has zero length.')
        x_local = vec_ij / L
        local_z_ref = el.get('local_z')
        use_default_orientation = True
        if local_z_ref is not None:
            local_z_ref = np.asarray(local_z_ref, dtype=float)
            if np.linalg.norm(local_z_ref) > 1e-09:
                if np.linalg.norm(np.cross(x_local, local_z_ref)) > 1e-09:
                    use_default_orientation = False
        if use_default_orientation:
            if np.abs(x_local[2]) > 0.99999:
                ref_vec = np.array([0.0, 1.0, 0.0])
            else:
                ref_vec = np.array([0.0, 0.0, 1.0])
            y_local = np.cross(ref_vec, x_local)
        else:
            y_local = np.cross(local_z_ref, x_local)
        y_local /= np.linalg.norm(y_local)
        z_local = np.cross(x_local, y_local)
        R = np.array([x_local, y_local, z_local])
        T = scipy.linalg.block_diag(R, R, R, R)
        k_e = local_elastic_stiffness_matrix_3D_beam(el['E'], el['nu'], el['A'], L, el['Iy'], el['Iz'], el['J'])
        K_e_global = T.T @ k_e @ T
        dof_indices = np.concatenate([np.arange(node_i_idx * 6, node_i_idx * 6 + 6), np.arange(node_j_idx * 6, node_j_idx * 6 + 6)])
        K[np.ix_(dof_indices, dof_indices)] += K_e_global
        element_data.append({'L': L, 'T': T, 'k_e': k_e, 'dof_indices': dof_indices})
    for (node_idx, load_vector) in nodal_loads.items():
        start_dof = node_idx * 6
        P[start_dof:start_dof + 6] += load_vector
    constrained_dofs = set()
    for (node_idx, bc_spec) in boundary_conditions.items():
        if not bc_spec:
            continue
        if isinstance(bc_spec[0], (bool, np.bool_)):
            constrained_local_dofs = [i for (i, constrained) in enumerate(bc_spec) if constrained]
        else:
            constrained_local_dofs = bc_spec
        for local_dof in constrained_local_dofs:
            constrained_dofs.add(node_idx * 6 + local_dof)
    all_dofs = np.arange(n_dof)
    free_dofs = np.setdiff1d(all_dofs, list(constrained_dofs), assume_unique=True)
    if len(free_dofs) == 0:
        raise ValueError('No free degrees of freedom.')
    K_reduced = K[np.ix_(free_dofs, free_dofs)]
    P_reduced = P[free_dofs]
    try:
        u_reduced = scipy.linalg.solve(K_reduced, P_reduced, assume_a='sym')
    except np.linalg.LinAlgError:
        raise ValueError('Reduced elastic stiffness matrix is singular. Check boundary conditions for rigid-body modes.')
    u = np.zeros(n_dof)
    u[free_dofs] = u_reduced
    K_g = np.zeros((n_dof, n_dof))
    for (i, el) in enumerate(elements):
        data = element_data[i]
        dof_indices = data['dof_indices']
        u_e_global = u[dof_indices]
        u_e_local = data['T'] @ u_e_global
        f_e_local = data['k_e'] @ u_e_local
        k_g_local = local_geometric_stiffness_matrix_3D_beam(data['L'], el['A'], el['I_rho'], Fx2=f_e_local[6], Mx2=f_e_local[9], My1=f_e_local[4], Mz1=f_e_local[5], My2=f_e_local[10], Mz2=f_e_local[11])
        K_g_global = data['T'].T @ k_g_local @ data['T']
        K_g[np.ix_(dof_indices, dof_indices)] += K_g_global
    K_g_reduced = K_g[np.ix_(free_dofs, free_dofs)]
    try:
        (eigenvalues, eigenvectors) = scipy.linalg.eigh(K_reduced, -K_g_reduced)
    except (np.linalg.LinAlgError, scipy.linalg.LinAlgError):
        raise ValueError('Eigenvalue problem failed to solve. Check for singularities in K or K_g.')
    positive_mask = eigenvalues > 1e-09
    if not np.any(positive_mask):
        raise ValueError('No positive eigenvalue found, buckling analysis failed.')
    positive_eigenvalues = eigenvalues[positive_mask]
    positive_eigenvectors = eigenvectors[:, positive_mask]
    min_idx = np.argmin(positive_eigenvalues)
    elastic_critical_load_factor = positive_eigenvalues[min_idx]
    mode_shape_reduced = positive_eigenvectors[:, min_idx]
    if np.iscomplexobj(elastic_critical_load_factor) or np.iscomplexobj(mode_shape_reduced):
        if np.max(np.abs(np.imag(elastic_critical_load_factor))) > 1e-09 or np.max(np.abs(np.imag(mode_shape_reduced))) > 1e-09:
            raise ValueError('Eigenvalue analysis resulted in significant complex numbers.')
        elastic_critical_load_factor = np.real(elastic_critical_load_factor)
        mode_shape_reduced = np.real(mode_shape_reduced)
    deformed_shape_vector = np.zeros(n_dof)
    deformed_shape_vector[free_dofs] = mode_shape_reduced
    return (float(elastic_critical_load_factor), deformed_shape_vector)