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
        All constrained DOFs are removed from the free set. It is the caller's
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
    n_dofs = 6 * n_nodes
    K_global = np.zeros((n_dofs, n_dofs))
    for element in elements:
        node_i = element['node_i']
        node_j = element['node_j']
        coord_i = node_coords[node_i]
        coord_j = node_coords[node_j]
        L = np.linalg.norm(coord_j - coord_i)
        E = element['E']
        nu = element['nu']
        A = element['A']
        I_y = element['I_y']
        I_z = element['I_z']
        J = element['J']
        k_local = local_elastic_stiffness_matrix_3D_beam(E, nu, A, L, I_y, I_z, J)
        local_z = element.get('local_z', None)
        T = _compute_transformation_matrix_3D(coord_i, coord_j, local_z)
        k_global = T.T @ k_local @ T
        dofs_i = range(6 * node_i, 6 * node_i + 6)
        dofs_j = range(6 * node_j, 6 * node_j + 6)
        all_dofs = np.concatenate([dofs_i, dofs_j])
        K_global[np.ix_(all_dofs, all_dofs)] += k_global
    P_global = np.zeros(n_dofs)
    for (node_idx, loads) in nodal_loads.items():
        start_dof = 6 * node_idx
        P_global[start_dof:start_dof + 6] = loads
    constrained_dofs = _parse_boundary_conditions(boundary_conditions, n_nodes)
    free_dofs = np.setdiff1d(np.arange(n_dofs), constrained_dofs)
    if len(free_dofs) == 0:
        raise ValueError('No free DOFs remaining after applying boundary conditions')
    K_free = K_global[np.ix_(free_dofs, free_dofs)]
    P_free = P_global[free_dofs]
    try:
        u_free = scipy.linalg.solve(K_free, P_free, assume_a='sym')
    except scipy.linalg.LinAlgError:
        raise ValueError('Linear system is singular - check boundary conditions')
    u_global = np.zeros(n_dofs)
    u_global[free_dofs] = u_free
    K_geo_global = np.zeros((n_dofs, n_dofs))
    for element in elements:
        node_i = element['node_i']
        node_j = element['node_j']
        coord_i = node_coords[node_i]
        coord_j = node_coords[node_j]
        L = np.linalg.norm(coord_j - coord_i)
        A = element['A']
        I_rho = element['I_rho']
        local_z = element.get('local_z', None)
        T = _compute_transformation_matrix_3D(coord_i, coord_j, local_z)
        dofs_i = range(6 * node_i, 6 * node_i + 6)
        dofs_j = range(6 * node_j, 6 * node_j + 6)
        all_dofs = np.concatenate([dofs_i, dofs_j])
        u_element_global = u_global[all_dofs]
        u_element_local = T @ u_element_global
        k_local = local_elastic_stiffness_matrix_3D_beam(element['E'], element['nu'], element['A'], L, element['I_y'], element['I_z'], element['J'])
        f_element_local = k_local @ u_element_local
        Fx2 = f_element_local[6]
        Mx2 = f_element_local[9]
        My1 = f_element_local[4]
        Mz1 = f_element_local[5]
        My2 = f_element_local[10]
        Mz2 = f_element_local[11]
        k_geo_local = local_geometric_stiffness_matrix_3D_beam(L, A, I_rho, Fx2, Mx2, My1, Mz1, My2, Mz2)
        k_geo_global_element = T.T @ k_geo_local @ T
        K_geo_global[np.ix_(all_dofs, all_dofs)] += k_geo_global_element
    K_free = K_global[np.ix_(free_dofs, free_dofs)]
    K_geo_free = K_geo_global[np.ix_(free_dofs, free_dofs)]
    try:
        (eigenvalues, eigenvectors) = scipy.linalg.eig(K_free, -K_geo_free)
    except scipy.linalg.LinAlgError:
        raise ValueError('Generalized eigenvalue problem failed to solve')
    real_mask = np.abs(eigenvalues.imag) < 1e-10
    eigenvalues_real = eigenvalues[real_mask].real
    eigenvectors_real = eigenvectors[:, real_mask].real
    positive_mask = eigenvalues_real > 1e-10
    if not np.any(positive_mask):
        raise ValueError('No positive eigenvalues found')
    min_positive_idx = np.argmin(eigenvalues_real[positive_mask])
    elastic_critical_load_factor = eigenvalues_real[positive_mask][min_positive_idx]
    eigenvector_free = eigenvectors_real[:, positive_mask][:, min_positive_idx]
    deformed_shape_vector = np.zeros(n_dofs)
    deformed_shape_vector[free_dofs] = eigenvector_free
    return (elastic_critical_load_factor, deformed_shape_vector)