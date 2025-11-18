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

    def create_local_coordinate_system(node_i_coords, node_j_coords, local_z_direction):
        local_x = node_j_coords - node_i_coords
        length = np.linalg.norm(local_x)
        if length < 1e-12:
            raise ValueError('Element has zero length')
        local_x = local_x / length
        if local_z_direction is None:
            if abs(local_x[2]) < 0.9:
                temp = np.array([0.0, 0.0, 1.0])
            else:
                temp = np.array([1.0, 0.0, 0.0])
            local_z = temp - np.dot(temp, local_x) * local_x
            local_z = local_z / np.linalg.norm(local_z)
        else:
            local_z = np.array(local_z_direction)
            local_z = local_z / np.linalg.norm(local_z)
            local_z = local_z - np.dot(local_z, local_x) * local_x
            local_z = local_z / np.linalg.norm(local_z)
        local_y = np.cross(local_z, local_x)
        return (local_x, local_y, local_z, length)

    def create_transformation_matrix(local_x, local_y, local_z):
        T = np.zeros((12, 12))
        R = np.column_stack([local_x, local_y, local_z])
        for i in range(4):
            T[3 * i:3 * i + 3, 3 * i:3 * i + 3] = R
        return T

    def create_element_elastic_stiffness(element, length):
        E = element['E']
        nu = element['nu']
        A = element['A']
        I_y = element['I_y']
        I_z = element['I_z']
        J = element['J']
        G = E / (2 * (1 + nu))
        k = np.zeros((12, 12))
        k[0, 0] = E * A / length
        k[0, 6] = -E * A / length
        k[6, 0] = -E * A / length
        k[6, 6] = E * A / length
        k[3, 3] = G * J / length
        k[3, 9] = -G * J / length
        k[9, 3] = -G * J / length
        k[9, 9] = G * J / length
        k[2, 2] = 12 * E * I_y / length ** 3
        k[2, 4] = 6 * E * I_y / length ** 2
        k[2, 8] = -12 * E * I_y / length ** 3
        k[2, 10] = 6 * E * I_y / length ** 2
        k[4, 2] = 6 * E * I_y / length ** 2
        k[4, 4] = 4 * E * I_y / length
        k[4, 8] = -6 * E * I_y / length ** 2
        k[4, 10] = 2 * E * I_y / length
        k[8, 2] = -12 * E * I_y / length ** 3
        k[8, 4] = -6 * E * I_y / length ** 2
        k[8, 8] = 12 * E * I_y / length ** 3
        k[8, 10] = -6 * E * I_y / length ** 2
        k[10, 2] = 6 * E * I_y / length ** 2
        k[10, 4] = 2 * E * I_y / length
        k[10, 8] = -6 * E * I_y / length ** 2
        k[10, 10] = 4 * E * I_y / length
        k[1, 1] = 12 * E * I_z / length ** 3
        k[1, 5] = -6 * E * I_z / length ** 2
        k[1, 7] = -12 * E * I_z / length ** 3
        k[1, 11] = -6 * E * I_z / length ** 2
        k[5, 1] = -6 * E * I_z / length ** 2
        k[5, 5] = 4 * E * I_z / length
        k[5, 7] = 6 * E * I_z / length ** 2
        k[5, 11] = 2 * E * I_z / length
        k[7, 1] = -12 * E * I_z / length ** 3
        k[7, 5] = 6 * E * I_z / length ** 2
        k[7, 7] = 12 * E * I_z / length ** 3
        k[7, 11] = 6 * E * I_z / length ** 2
        k[11, 1] = -6 * E * I_z / length ** 2
        k[11, 5] = 2 * E * I_z / length
        k[11, 7] = 6 * E * I_z / length ** 2
        k[11, 11] = 4 * E * I_z / length
        return k

    def create_element_geometric_stiffness(element, length, axial_force):
        I_y = element['I_y']
        I_z = element['I_z']
        I_rho = element['I_rho']
        kg = np.zeros((12, 12))
        if abs(axial_force) < 1e-12:
            return kg
        kg[2, 2] = 6 / 5 * axial_force / length
        kg[2, 8] = -6 / 5 * axial_force / length
        kg[8, 2] = -6 / 5 * axial_force / length
        kg[8, 8] = 6 / 5 * axial_force / length
        kg[4, 4] = 2 * axial_force * length / 15
        kg[4, 10] = -axial_force * length / 30
        kg[10, 4] = -axial_force * length / 30
        kg[10, 10] = 2 * axial_force * length / 15
        kg[2, 4] = axial_force / 10
        kg[2, 10] = -axial_force / 10
        kg[4, 2] = axial_force / 10
        kg[10, 2] = -axial_force / 10
        kg[8, 4] = -axial_force / 10
        kg[8, 10] = axial_force / 10
        kg[4, 8] = -axial_force / 10
        kg[10, 8] = axial_force / 10
        kg[1, 1] = 6 / 5 * axial_force / length
        kg[1, 7] = -6 / 5 * axial_force / length
        kg[7, 1] = -6 / 5 * axial_force / length
        kg[7, 7] = 6 / 5 * axial_force / length
        kg[5, 5] = 2 * axial_force * length / 15
        kg[5, 11] = -axial_force * length / 30
        kg[11, 5] = -axial_force * length / 30
        kg[11, 11] = 2 * axial_force * length / 15
        kg[1, 5] = -axial_force / 10
        kg[1, 11] = axial_force / 10
        kg[5, 1] = -axial_force / 10
        kg[11, 1] = axial_force / 10
        kg[7, 5] = axial_force / 10
        kg[7, 11] = -axial_force / 10
        kg[5, 7] = axial_force / 10
        kg[11, 7] = -axial_force / 10
        return kg
    K = np.zeros((n_dofs, n_dofs))
    for element in elements:
        node_i = element['node_i']
        node_j = element['node_j']
        node_i_coords = node_coords[node_i]
        node_j_coords = node_coords[node_j]
        local_z_direction = element.get('local_z', None)
        (local_x, local_y, local_z, length) = create_local_coordinate_system(node_i_coords, node_j_coords, local_z_direction)
        T = create_transformation_matrix(local_x, local_y, local_z)
        k_local = create_element_elastic_stiffness(element, length)
        k_global = T.T @ k_local @ T
        dofs = np.concatenate([np.arange(6 * node_i, 6 * node_i + 6), np.arange(6 * node_j, 6 * node_j + 6)])
        for i in range(12):
            for j in range(12):
                K[dofs[i], dofs[j]] += k_global[i, j]
    P = np.zeros(n_dofs)
    for (node_idx, loads) in nodal_loads.items():
        P[6 * node_idx:6 * node_idx + 6] = loads
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
    K_free = K[np.ix_(free_dofs, free_dofs)]
    P_free = P[free_dofs]
    try:
        u_free = np.linalg.solve(K_free, P_free)
    except np.linalg.LinAlgError:
        raise ValueError('Singular stiffness matrix')
    u_global = np.zeros(n_dofs)
    u_global[free_dofs] = u_free
    Kg = np.zeros((n_dofs, n_dofs))
    for element in elements:
        node_i = element['node_i']
        node_j = element['node_j']
        node_i_coords = node_coords[node_i]
        node_j_coords = node_coords[node_j]
        local_z_direction = element.get('local_z', None)
        (local_x, local_y, local_z, length) = create_local_coordinate_system(node_i_coords, node_j_coords, local_z_direction)
        T = create_transformation_matrix(local_x, local_y, local_z)
        dofs = np.concatenate([np.arange(6 * node_i, 6 * node_i + 6), np.arange(6 * node_j, 6 * node_j + 6)])
        u_element = u_global[dofs]
        u_local = T @ u_element
        E = element['E']
        A = element['A']
        axial_force = E * A * (u_local[6] - u_local[0]) / length
        kg_local = create_element_geometric_stiffness(element, length, axial_force)
        kg_global = T.T @ kg_local @ T
        for i in range(12):
            for j in range(12):
                Kg[dofs[i], dofs[j]] += kg_global[i, j]
    K_free = K[np.ix_(free_dofs, free_dofs)]
    Kg_free = Kg[np.ix_(free_dofs, free_dofs)]
    try:
        (eigenvalues, eigenvectors) = scipy.linalg.eigh(K_free, -Kg_free)
    except np.linalg.LinAlgError:
        raise ValueError('Failed to solve eigenvalue problem')
    positive_eigenvalues = eigenvalues[eigenvalues > 1e-08]
    if len(positive_eigenvalues) == 0:
        raise ValueError('No positive eigenvalues found')
    min_idx = np.argmin(positive_eigenvalues)
    critical_load_factor = positive_eigenvalues[min_idx]
    eigenvalue_idx = np.where(eigenvalues == positive_eigenvalues[min_idx])[0][0]
    mode_shape_free = eigenvectors[:, eigenvalue_idx]
    mode_shape_global = np.zeros(n_dofs)
    mode_shape_global[free_dofs] = mode_shape_free
    return (critical_load_factor, mode_shape_global)