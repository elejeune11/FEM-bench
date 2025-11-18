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

    def compute_transformation_matrix(node_i_coords, node_j_coords, local_z):
        dx = node_j_coords - node_i_coords
        L = np.linalg.norm(dx)
        if L < 1e-12:
            raise ValueError('Element has zero length')
        local_x = dx / L
        if local_z is None:
            if abs(local_x[2]) < 0.9:
                temp = np.array([0.0, 0.0, 1.0])
            else:
                temp = np.array([1.0, 0.0, 0.0])
            local_y = np.cross(temp, local_x)
            local_y = local_y / np.linalg.norm(local_y)
            local_z = np.cross(local_x, local_y)
        else:
            local_z = np.array(local_z)
            local_z = local_z / np.linalg.norm(local_z)
            local_y = np.cross(local_z, local_x)
            local_y = local_y / np.linalg.norm(local_y)
            local_z = np.cross(local_x, local_y)
        R = np.array([local_x, local_y, local_z])
        T = np.zeros((12, 12))
        for i in range(4):
            T[3 * i:3 * i + 3, 3 * i:3 * i + 3] = R
        return (T, L)

    def compute_element_elastic_stiffness_local(element, L):
        E = element['E']
        nu = element['nu']
        A = element['A']
        Iy = element['Iy']
        Iz = element['Iz']
        J = element['J']
        G = E / (2 * (1 + nu))
        k = np.zeros((12, 12))
        k[0, 0] = E * A / L
        k[0, 6] = -E * A / L
        k[6, 0] = -E * A / L
        k[6, 6] = E * A / L
        k[3, 3] = G * J / L
        k[3, 9] = -G * J / L
        k[9, 3] = -G * J / L
        k[9, 9] = G * J / L
        k[2, 2] = 12 * E * Iy / L ** 3
        k[2, 4] = 6 * E * Iy / L ** 2
        k[2, 8] = -12 * E * Iy / L ** 3
        k[2, 10] = 6 * E * Iy / L ** 2
        k[4, 2] = 6 * E * Iy / L ** 2
        k[4, 4] = 4 * E * Iy / L
        k[4, 8] = -6 * E * Iy / L ** 2
        k[4, 10] = 2 * E * Iy / L
        k[8, 2] = -12 * E * Iy / L ** 3
        k[8, 4] = -6 * E * Iy / L ** 2
        k[8, 8] = 12 * E * Iy / L ** 3
        k[8, 10] = -6 * E * Iy / L ** 2
        k[10, 2] = 6 * E * Iy / L ** 2
        k[10, 4] = 2 * E * Iy / L
        k[10, 8] = -6 * E * Iy / L ** 2
        k[10, 10] = 4 * E * Iy / L
        k[1, 1] = 12 * E * Iz / L ** 3
        k[1, 5] = -6 * E * Iz / L ** 2
        k[1, 7] = -12 * E * Iz / L ** 3
        k[1, 11] = -6 * E * Iz / L ** 2
        k[5, 1] = -6 * E * Iz / L ** 2
        k[5, 5] = 4 * E * Iz / L
        k[5, 7] = 6 * E * Iz / L ** 2
        k[5, 11] = 2 * E * Iz / L
        k[7, 1] = -12 * E * Iz / L ** 3
        k[7, 5] = 6 * E * Iz / L ** 2
        k[7, 7] = 12 * E * Iz / L ** 3
        k[7, 11] = 6 * E * Iz / L ** 2
        k[11, 1] = -6 * E * Iz / L ** 2
        k[11, 5] = 2 * E * Iz / L
        k[11, 7] = 6 * E * Iz / L ** 2
        k[11, 11] = 4 * E * Iz / L
        return k

    def compute_element_geometric_stiffness_local(element, L, N):
        Iy = element['Iy']
        Iz = element['Iz']
        I_rho = element['I_rho']
        A = element['A']
        kg = np.zeros((12, 12))
        if abs(N) > 1e-12:
            kg[1, 1] = 6 / 5 * N / L
            kg[1, 5] = N / 10
            kg[1, 7] = -6 / 5 * N / L
            kg[1, 11] = N / 10
            kg[5, 1] = N / 10
            kg[5, 5] = 2 * N * L / 15
            kg[5, 7] = -N / 10
            kg[5, 11] = -N * L / 30
            kg[7, 1] = -6 / 5 * N / L
            kg[7, 5] = -N / 10
            kg[7, 7] = 6 / 5 * N / L
            kg[7, 11] = -N / 10
            kg[11, 1] = N / 10
            kg[11, 5] = -N * L / 30
            kg[11, 7] = -N / 10
            kg[11, 11] = 2 * N * L / 15
            kg[2, 2] = 6 / 5 * N / L
            kg[2, 4] = -N / 10
            kg[2, 8] = -6 / 5 * N / L
            kg[2, 10] = -N / 10
            kg[4, 2] = -N / 10
            kg[4, 4] = 2 * N * L / 15
            kg[4, 8] = N / 10
            kg[4, 10] = -N * L / 30
            kg[8, 2] = -6 / 5 * N / L
            kg[8, 4] = N / 10
            kg[8, 8] = 6 / 5 * N / L
            kg[8, 10] = N / 10
            kg[10, 2] = -N / 10
            kg[10, 4] = -N * L / 30
            kg[10, 8] = N / 10
            kg[10, 10] = 2 * N * L / 15
            if I_rho > 0:
                factor = N * I_rho / A
                kg[3, 3] += factor / L
                kg[3, 9] += -factor / L
                kg[9, 3] += -factor / L
                kg[9, 9] += factor / L
        return kg
    K_global = np.zeros((n_dofs, n_dofs))
    for element in elements:
        node_i = element['node_i']
        node_j = element['node_j']
        node_i_coords = node_coords[node_i]
        node_j_coords = node_coords[node_j]
        local_z = element.get('local_z', None)
        (T, L) = compute_transformation_matrix(node_i_coords, node_j_coords, local_z)
        k_local = compute_element_elastic_stiffness_local(element, L)
        k_global = T.T @ k_local @ T
        dofs = np.concatenate([np.arange(6 * node_i, 6 * node_i + 6), np.arange(6 * node_j, 6 * node_j + 6)])
        for i in range(12):
            for j in range(12):
                K_global[dofs[i], dofs[j]] += k_global[i, j]
    P_global = np.zeros(n_dofs)
    for (node_idx, loads) in nodal_loads.items():
        P_global[6 * node_idx:6 * node_idx + 6] = loads
    constrained_dofs = set()
    for (node_idx, bc_spec) in boundary_conditions.items():
        if len(bc_spec) == 6 and all((isinstance(x, (bool, np.bool_)) for x in bc_spec)):
            for (dof_local, is_constrained) in enumerate(bc_spec):
                if is_constrained:
                    constrained_dofs.add(6 * node_idx + dof_local)
        else:
            for dof_local in bc_spec:
                constrained_dofs.add(6 * node_idx + dof_local)
    free_dofs = np.array([i for i in range(n_dofs) if i not in constrained_dofs])
    if len(free_dofs) == 0:
        raise ValueError('No free DOFs available')
    K_free = K_global[np.ix_(free_dofs, free_dofs)]
    P_free = P_global[free_dofs]
    try:
        u_free = scipy.linalg.solve(K_free, P_free)
    except scipy.linalg.LinAlgError:
        raise ValueError('Singular stiffness matrix')
    u_global = np.zeros(n_dofs)
    u_global[free_dofs] = u_free
    Kg_global = np.zeros((n_dofs, n_dofs))
    for element in elements:
        node_i = element['node_i']
        node_j = element['node_j']
        node_i_coords = node_coords[node_i]
        node_j_coords = node_coords[node_j]
        local_z = element.get('local_z', None)
        (T, L) = compute_transformation_matrix(node_i_coords, node_j_coords, local_z)
        dofs = np.concatenate([np.arange(6 * node_i, 6 * node_i + 6), np.arange(6 * node_j, 6 * node_j + 6)])
        u_element_global = u_global[dofs]
        u_element_local = T @ u_element_global
        E = element['E']
        A = element['A']
        N = E * A * (u_element_local[6] - u_element_local[0]) / L
        kg_local = compute_element_geometric_stiffness_local(element, L, N)
        kg_global = T.T @ kg_local @ T
        for i in range(12):
            for j in range(12):
                Kg_global[dofs[i], dofs[j]] += kg_global[i, j]
    K_free = K_global[np.ix_(free_dofs, free_dofs)]
    Kg_free = Kg_global[np.ix_(free_dofs, free_dofs)]
    try:
        (eigenvals, eigenvecs) = scipy.linalg.eigh(K_free, -Kg_free)
    except scipy.linalg.LinAlgError:
        raise ValueError('Failed to solve eigenvalue problem')
    positive_eigenvals = eigenvals[eigenvals > 1e-12]
    if len(positive_eigenvals) == 0:
        raise ValueError('No positive eigenvalues found')
    min_eigenval = np.min(positive_eigenvals)
    min_idx = np.where(eigenvals == min_eigenval)[0][0]
    mode_shape_global = np.zeros(n_dofs)
    mode_shape_global[free_dofs] = eigenvecs[:, min_idx]
    return (min_eigenval, mode_shape_global)