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
        node's specification can be provided in either of two forms:
          the DOF is constrained (fixed).
          at that node are fixed.
        All constrained DOFs are removed from the free set. It is the caller's
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

    def compute_transformation_matrix(node_i_coords, node_j_coords, local_z_dir=None):
        dx = node_j_coords - node_i_coords
        L = np.linalg.norm(dx)
        if L < 1e-12:
            raise ValueError('Element has zero length')
        ex = dx / L
        if local_z_dir is not None:
            ez = np.array(local_z_dir, dtype=float)
            ez = ez / np.linalg.norm(ez)
        elif abs(np.dot(ex, [0, 0, 1])) > 0.9:
            ez = np.array([1.0, 0.0, 0.0])
        else:
            ez = np.array([0.0, 0.0, 1.0])
        ey = np.cross(ez, ex)
        ey = ey / np.linalg.norm(ey)
        ez = np.cross(ex, ey)
        R = np.array([ex, ey, ez]).T
        T = np.zeros((12, 12))
        for i in range(4):
            T[3 * i:3 * i + 3, 3 * i:3 * i + 3] = R
        return (T, L)

    def compute_element_stiffness_local(E, nu, A, Iy, Iz, J, L):
        G = E / (2 * (1 + nu))
        k = np.zeros((12, 12))
        k[0, 0] = k[6, 6] = E * A / L
        k[0, 6] = k[6, 0] = -E * A / L
        EIz_L3 = E * Iz / L ** 3
        k[2, 2] = k[8, 8] = 12 * EIz_L3
        k[2, 8] = k[8, 2] = -12 * EIz_L3
        k[2, 4] = k[4, 2] = k[8, 10] = k[10, 8] = 6 * EIz_L3 * L
        k[2, 10] = k[10, 2] = k[4, 8] = k[8, 4] = -6 * EIz_L3 * L
        k[4, 4] = k[10, 10] = 4 * EIz_L3 * L ** 2
        k[4, 10] = k[10, 4] = 2 * EIz_L3 * L ** 2
        EIy_L3 = E * Iy / L ** 3
        k[1, 1] = k[7, 7] = 12 * EIy_L3
        k[1, 7] = k[7, 1] = -12 * EIy_L3
        k[1, 5] = k[5, 1] = k[7, 11] = k[11, 7] = -6 * EIy_L3 * L
        k[1, 11] = k[11, 1] = k[5, 7] = k[7, 5] = 6 * EIy_L3 * L
        k[5, 5] = k[11, 11] = 4 * EIy_L3 * L ** 2
        k[5, 11] = k[11, 5] = 2 * EIy_L3 * L ** 2
        k[3, 3] = k[9, 9] = G * J / L
        k[3, 9] = k[9, 3] = -G * J / L
        return k

    def compute_element_geometric_stiffness_local(N, L, I_rho):
        kg = np.zeros((12, 12))
        if abs(N) < 1e-12:
            return kg
        c1 = N / L
        c2 = N / (30 * L)
        c3 = N / (6 * L)
        kg[1, 1] = kg[7, 7] = c1 * 6 / 5
        kg[1, 7] = kg[7, 1] = -c1 * 6 / 5
        kg[2, 2] = kg[8, 8] = c1 * 6 / 5
        kg[2, 8] = kg[8, 2] = -c1 * 6 / 5
        kg[1, 5] = kg[5, 1] = kg[7, 11] = kg[11, 7] = c1 * L / 10
        kg[1, 11] = kg[11, 1] = kg[5, 7] = kg[7, 5] = -c1 * L / 10
        kg[2, 4] = kg[4, 2] = kg[8, 10] = kg[10, 8] = -c1 * L / 10
        kg[2, 10] = kg[10, 2] = kg[4, 8] = kg[8, 4] = c1 * L / 10
        kg[5, 5] = kg[11, 11] = c1 * L ** 2 * 2 / 15
        kg[5, 11] = kg[11, 5] = -c1 * L ** 2 / 30
        kg[4, 4] = kg[10, 10] = c1 * L ** 2 * 2 / 15
        kg[4, 10] = kg[10, 4] = -c1 * L ** 2 / 30
        if I_rho > 0:
            kt = N * I_rho / L
            kg[3, 3] = kg[9, 9] = kt
            kg[3, 9] = kg[9, 3] = -kt
        return kg
    K_global = np.zeros((n_dofs, n_dofs))
    for elem in elements:
        node_i = elem['node_i']
        node_j = elem['node_j']
        node_i_coords = node_coords[node_i]
        node_j_coords = node_coords[node_j]
        local_z = elem.get('local_z', None)
        (T, L) = compute_transformation_matrix(node_i_coords, node_j_coords, local_z)
        k_local = compute_element_stiffness_local(elem['E'], elem['nu'], elem['A'], elem['Iy'], elem['Iz'], elem['J'], L)
        k_global = T.T @ k_local @ T
        dofs_i = np.arange(6 * node_i, 6 * node_i + 6)
        dofs_j = np.arange(6 * node_j, 6 * node_j + 6)
        dofs = np.concatenate([dofs_i, dofs_j])
        for (i, dof_i) in enumerate(dofs):
            for (j, dof_j) in enumerate(dofs):
                K_global[dof_i, dof_j] += k_global[i, j]
    P_global = np.zeros(n_dofs)
    for (node_idx, loads) in nodal_loads.items():
        dofs = np.arange(6 * node_idx, 6 * node_idx + 6)
        P_global[dofs] += loads
    constrained_dofs = set()
    for (node_idx, bc_spec) in boundary_conditions.items():
        if len(bc_spec) == 6 and all((isinstance(x, (bool, np.bool_)) for x in bc_spec)):
            for (i, is_constrained) in enumerate(bc_spec):
                if is_constrained:
                    constrained_dofs.add(6 * node_idx + i)
        else:
            for dof_idx in bc_spec:
                constrained_dofs.add(6 * node_idx + dof_idx)
    free_dofs = np.array([i for i in range(n_dofs) if i not in constrained_dofs])
    if len(free_dofs) == 0:
        raise ValueError('No free DOFs available')
    K_free = K_global[np.ix_(free_dofs, free_dofs)]
    P_free = P_global[free_dofs]
    try:
        u_free = np.linalg.solve(K_free, P_free)
    except np.linalg.LinAlgError:
        raise ValueError('Singular stiffness matrix - check boundary conditions')
    u_global = np.zeros(n_dofs)
    u_global[free_dofs] = u_free
    Kg_global = np.zeros((n_dofs, n_dofs))
    for elem in elements:
        node_i = elem['node_i']
        node_j = elem['node_j']
        node_i_coords = node_coords[node_i]
        node_j_coords = node_coords[node_j]
        local_z = elem.get('local_z', None)
        (T, L) = compute_transformation_matrix(node_i_coords, node_j_coords, local_z)
        dofs_i = np.arange(6 * node_i, 6 * node_i + 6)
        dofs_j = np.arange(6 * node_j, 6 * node_j + 6)
        dofs = np.concatenate([dofs_i, dofs_j])
        u_elem_global = u_global[dofs]
        u_elem_local = T @ u_elem_global
        E = elem['E']
        A = elem['A']
        N = E * A * (u_elem_local[6] - u_elem_local[0]) / L
        kg_local = compute_element_geometric_stiffness_local(N, L, elem['I_rho'])
        kg_global = T.T @ kg_local @ T
        for (i, dof_i) in enumerate(dofs):
            for (j, dof_j) in enumerate(dofs):
                Kg_global[dof_i, dof_j] += kg_global[i, j]
    Kg_free = Kg_global[np.ix_(free_dofs, free_dofs)]
    try:
        (eigenvals, eigenvecs) = scipy.linalg.eigh(K_free, -Kg_free)
    except np.linalg.LinAlgError:
        raise ValueError('Failed to solve eigenvalue problem')
    positive_eigenvals = eigenvals[eigenvals > 1e-06]
    if len(positive_eigenvals) == 0:
        raise ValueError('No positive eigenvalues found')
    min_eigenval = np.min(positive_eigenvals)
    min_idx = np.where(eigenvals == min_eigenval)[0][0]
    mode_free = eigenvecs[:, min_idx]
    mode_global = np.zeros(n_dofs)
    mode_global[free_dofs] = mode_free
    return (min_eigenval, mode_global)