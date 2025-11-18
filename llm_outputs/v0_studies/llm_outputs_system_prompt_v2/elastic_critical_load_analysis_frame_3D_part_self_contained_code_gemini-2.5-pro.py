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
    n_dof = 6 * n_nodes
    K = np.zeros((n_dof, n_dof))
    K_g = np.zeros((n_dof, n_dof))
    P = np.zeros(n_dof)
    for (node_idx, loads) in nodal_loads.items():
        start_dof = 6 * node_idx
        P[start_dof:start_dof + 6] = loads
    element_data = []
    for elem in elements:
        (i, j) = (elem['node_i'], elem['node_j'])
        coord_i = node_coords[i]
        coord_j = node_coords[j]
        vec_ij = coord_j - coord_i
        L = np.linalg.norm(vec_ij)
        if L < 1e-12:
            continue
        x_vec = vec_ij / L
        use_default_orientation = True
        local_z_dir = elem.get('local_z')
        if local_z_dir is not None:
            local_z_dir_vec = np.array(local_z_dir, dtype=float)
            if np.linalg.norm(local_z_dir_vec) > 1e-09 and np.linalg.norm(np.cross(x_vec, local_z_dir_vec)) > 1e-06:
                z_global = local_z_dir_vec / np.linalg.norm(local_z_dir_vec)
                y_vec = np.cross(z_global, x_vec)
                y_vec /= np.linalg.norm(y_vec)
                z_vec = np.cross(x_vec, y_vec)
                use_default_orientation = False
        if use_default_orientation:
            ref_vec = np.array([0.0, 0.0, 1.0])
            if np.linalg.norm(np.cross(x_vec, ref_vec)) < 1e-06:
                ref_vec = np.array([0.0, 1.0, 0.0])
            y_vec = np.cross(ref_vec, x_vec)
            y_vec /= np.linalg.norm(y_vec)
            z_vec = np.cross(x_vec, y_vec)
        R = np.vstack([x_vec, y_vec, z_vec])
        T_block = np.kron(np.eye(4, dtype=float), R)
        k_e = local_elastic_stiffness_matrix_3D_beam(E=elem['E'], nu=elem['nu'], A=elem['A'], L=L, Iy=elem['Iy'], Iz=elem['Iz'], J=elem['J'])
        K_e_global = T_block.T @ k_e @ T_block
        dof_indices = np.concatenate([np.arange(6 * i, 6 * i + 6), np.arange(6 * j, 6 * j + 6)])
        K[np.ix_(dof_indices, dof_indices)] += K_e_global
        element_data.append({'k_e': k_e, 'T_block': T_block, 'dof_indices': dof_indices, 'L': L, 'A': elem['A'], 'I_rho': elem['I_rho']})
    constrained_dofs = set()
    for (node_idx, bc_spec) in boundary_conditions.items():
        if not bc_spec:
            continue
        if isinstance(bc_spec[0], (bool, np.bool_)):
            for (i, is_constrained) in enumerate(bc_spec):
                if is_constrained:
                    constrained_dofs.add(6 * node_idx + i)
        else:
            for dof_local_idx in bc_spec:
                constrained_dofs.add(6 * node_idx + dof_local_idx)
    free_dofs = sorted(list(set(range(n_dof)) - constrained_dofs))
    if not free_dofs:
        raise ValueError('All DOFs are constrained.')
    K_free = K[np.ix_(free_dofs, free_dofs)]
    P_free = P[free_dofs]
    try:
        u_free = scipy.linalg.solve(K_free, P_free, assume_a='sym')
    except np.linalg.LinAlgError:
        raise ValueError('Singular elastic stiffness matrix. Check boundary conditions for rigid-body modes.')
    u = np.zeros(n_dof)
    u[free_dofs] = u_free
    for elem_data in element_data:
        dof_indices = elem_data['dof_indices']
        T_block = elem_data['T_block']
        k_e = elem_data['k_e']
        u_e_global = u[dof_indices]
        u_e_local = T_block @ u_e_global
        f_e_local = k_e @ u_e_local
        k_g = local_geometric_stiffness_matrix_3D_beam(L=elem_data['L'], A=elem_data['A'], I_rho=elem_data['I_rho'], Fx2=f_e_local[6], Mx2=f_e_local[9], My1=f_e_local[4], Mz1=f_e_local[5], My2=f_e_local[10], Mz2=f_e_local[11])
        K_g_global = T_block.T @ k_g @ T_block
        K_g[np.ix_(dof_indices, dof_indices)] += K_g_global
    K_g_free = K_g[np.ix_(free_dofs, free_dofs)]
    try:
        (eigenvalues, eigenvectors) = scipy.linalg.eigh(K_free, -K_g_free)
    except (np.linalg.LinAlgError, scipy.linalg.LinAlgError):
        raise ValueError('Eigenvalue problem failed to converge. Check model definition and loading.')
    if np.iscomplexobj(eigenvalues):
        if np.any(np.abs(np.imag(eigenvalues)) > 1e-09 * np.abs(np.real(eigenvalues))):
            raise ValueError('Significant complex part found in eigenvalues, indicating a problem with the model matrices.')
        eigenvalues = np.real(eigenvalues)
    positive_mask = eigenvalues > 1e-09
    if not np.any(positive_mask):
        raise ValueError('No positive eigenvalue found. The structure may be unstable under the reference load or the load does not cause buckling.')
    elastic_critical_load_factor = np.min(eigenvalues[positive_mask])
    min_lambda_idx = np.where(eigenvalues == elastic_critical_load_factor)[0][0]
    mode_shape_free = eigenvectors[:, min_lambda_idx]
    deformed_shape_vector = np.zeros(n_dof)
    deformed_shape_vector[free_dofs] = mode_shape_free
    return (elastic_critical_load_factor, deformed_shape_vector)