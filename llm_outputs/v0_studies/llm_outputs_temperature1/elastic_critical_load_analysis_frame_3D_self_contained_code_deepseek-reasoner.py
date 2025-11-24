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
    n_nodes = node_coords.shape[0]
    n_dofs = 6 * n_nodes
    fixed_dofs = set()
    for (node_idx, bc_spec) in boundary_conditions.items():
        if len(bc_spec) == 6 and all((isinstance(x, bool) for x in bc_spec)):
            for (dof_local, is_fixed) in enumerate(bc_spec):
                if is_fixed:
                    fixed_dofs.add(6 * node_idx + dof_local)
        else:
            for dof_local in bc_spec:
                fixed_dofs.add(6 * node_idx + dof_local)
    free_dofs = sorted(set(range(n_dofs)) - fixed_dofs)
    if len(free_dofs) == 0:
        raise ValueError('No free DOFs remain after applying boundary conditions')
    K = assemble_global_stiffness_matrix_linear_elastic_3D(node_coords, elements)
    P = assemble_global_load_vector_linear_elastic_3D(node_coords, nodal_loads)
    K_red = K[np.ix_(free_dofs, free_dofs)]
    P_red = P[free_dofs]
    try:
        u_red = scipy.linalg.solve(K_red, P_red, assume_a='sym')
    except scipy.linalg.LinAlgError as e:
        raise ValueError(f'Linear static solution failed: {e}') from e
    u = np.zeros(n_dofs)
    u[free_dofs] = u_red
    K_g = assemble_geometric_stiffness_matrix_3D(node_coords, elements, u)
    K_g_red = K_g[np.ix_(free_dofs, free_dofs)]
    try:
        (eigenvalues, eigenvectors_red) = scipy.linalg.eig(K_red, -K_g_red)
    except scipy.linalg.LinAlgError as e:
        raise ValueError(f'Eigenvalue solution failed: {e}') from e
    eigenvalues = np.real(eigenvalues)
    eigenvectors_red = np.real(eigenvectors_red)
    if np.any(np.abs(np.imag(eigenvalues)) > 1e-10):
        raise ValueError('Significant complex eigenvalues detected')
    positive_mask = eigenvalues > 0
    if not np.any(positive_mask):
        raise ValueError('No positive eigenvalues found')
    min_positive_idx = np.argmin(eigenvalues[positive_mask])
    elastic_critical_load_factor = eigenvalues[positive_mask][min_positive_idx]
    eigenvector_red = eigenvectors_red[:, positive_mask][:, min_positive_idx]
    deformed_shape_vector = np.zeros(n_dofs)
    deformed_shape_vector[free_dofs] = eigenvector_red
    return (elastic_critical_load_factor, deformed_shape_vector)