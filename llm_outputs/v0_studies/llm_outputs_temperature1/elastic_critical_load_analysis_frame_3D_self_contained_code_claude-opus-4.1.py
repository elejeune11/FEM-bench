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
    n_dof = 6 * n_nodes
    K = np.zeros((n_dof, n_dof))
    for elem in elements:
        node_i = elem['node_i']
        node_j = elem['node_j']
        L = np.linalg.norm(node_coords[node_j] - node_coords[node_i])
        E = elem['E']
        nu = elem['nu']
        A = elem['A']
        I_y = elem['I_y']
        I_z = elem['I_z']
        J = elem['J']
        G = E / (2 * (1 + nu))
        k_local = np.zeros((12, 12))
        k_local[0, 0] = k_local[6, 6] = E * A / L
        k_local[0, 6] = k_local[6, 0] = -E * A / L
        k_local[3, 3] = k_local[9, 9] = G * J / L
        k_local[3, 9] = k_local[9, 3] = -G * J / L
        k_local[2, 2] = k_local[8, 8] = 12 * E * I_y / L ** 3
        k_local[2, 4] = k_local[4, 2] = 6 * E * I_y / L ** 2
        k_local[2, 8] = k_local[8, 2] = -12 * E * I_y / L ** 3
        k_local[2, 10] = k_local[10, 2] = 6 * E * I_y / L ** 2
        k_local[4, 4] = 4 * E * I_y / L
        k_local[4, 8] = k_local[8, 4] = -6 * E * I_y / L ** 2
        k_local[4, 10] = k_local[10, 4] = 2 * E * I_y / L
        k_local[8, 10] = k_local[10, 8] = -6 * E * I_y / L ** 2
        k_local[10, 10] = 4 * E * I_y / L
        k_local[1, 1] = k_local[7, 7] = 12 * E * I_z / L ** 3
        k_local[1, 5] = k_local[5, 1] = -6 * E * I_z / L ** 2
        k_local[1, 7] = k_local[7, 1] = -12 * E * I_z / L ** 3
        k_local[1, 11] = k_local[11, 1] = -6 * E * I_z / L ** 2
        k_local[5, 5] = 4 * E * I_z / L
        k_local[5, 7] = k_local[7, 5] = 6 * E * I_z / L ** 2
        k_local[5, 11] = k_local[11, 5] = 2 * E * I_z / L
        k_local[7, 11] = k_local[11, 7] = 6 * E * I_z / L ** 2
        k_local[11, 11] = 4 * E * I_z / L
        x_vec = (node_coords[node_j] - node_coords[node_i]) / L
        if 'local_z' in elem and elem['local_z'] is not None:
            z_vec = np.array(elem['local_z'])
            z_vec = z_vec / np.linalg.norm(z_vec)
            y_vec = np.cross(z_vec, x_vec)
            y_vec = y_vec / np.linalg.norm(y_vec)
            z_vec = np.cross(x_vec, y_vec)
        else:
            if abs(x_vec[2]) < 0.99:
                z_vec = np.array([0, 0, 1])
            else:
                z_vec = np.array([1, 0, 0])
            y_vec = np.cross(z_vec, x_vec)
            y_vec = y_vec / np.linalg.norm(y_vec)
            z_vec = np.cross(x_vec, y_vec)
        T_elem = np.zeros((3, 3))
        T_elem[0, :] = x_vec
        T_elem[1, :] = y_vec
        T_elem[2, :] = z_vec
        T = np.zeros((12, 12))
        for i in range(4):
            T[3 * i:3 * i + 3, 3 * i:3 * i + 3] = T_elem
        k_global = T.T @ k_local @ T
        dof_i = [6 * node_i + j for j in range(6)]
        dof_j = [6 * node_j + j for j in range(6)]
        dofs = dof_i + dof_j
        for i in range(12):
            for j in range(12):
                K[dofs[i], dofs[j]] += k_global[i, j]
    P = np.zeros(n_dof)
    for (node_idx, loads) in nodal_loads.items():
        for i in range(6):
            P[6 * node_idx + i] = loads[i]
    constrained_dofs = set()
    for (node_idx, bc) in boundary_conditions.items():
        if all((isinstance(b, bool) for b in bc)):
            for (i, is_fixed) in enumerate(bc):
                if is_fixed:
                    constrained_dofs.add(6 * node_idx + i)
        else:
            for dof_local in bc:
                constrained_dofs.add(6 * node_idx + dof_local)
    free_dofs = [i for i in range(n_dof) if i not in constrained_dofs]
    n_free = len(free_dofs)
    K_ff = np.zeros((n_free, n_free))
    P_f = np.zeros(n_free)
    for (i, dof_i) in enumerate(free_dofs):
        P_f[i] = P[dof_i]
        for (j, dof_j) in enumerate(free_dofs):
            K_ff[i, j] = K[dof_i, dof_j]
    u_f = np.linalg.solve(K_ff, P_f)
    u = np.zeros(n_dof)
    for (i, dof) in enumerate(free_dofs):
        u[dof] = u_f[i]
    K_g = np.zeros((n_dof, n_dof))
    for elem in elements:
        node_i = elem['node_i']
        node_j = elem['node_j']
        L = np.linalg.norm(node_coords[node_j] - node_coords[node_i])
        dof_i = [6 * node_i + j for j in range(6)]
        dof_j = [6 * node_j + j for j in range(6)]
        dofs = dof_i + dof_j
        u_elem_global = np.array([u[dof] for dof in dofs])
        x_vec = (node_coords[node_j] - node_coords[node_i]) / L
        if 'local_z' in elem and elem['local_z'] is not None:
            z_vec = np.array(elem['local_z'])
            z_vec = z_vec / np.linalg.norm(z_vec)
            y_vec = np.cross(z_vec, x_vec)
            y_vec = y_vec / np.linalg.norm(y_vec)
            z_vec = np.cross(x_vec, y_vec)
        else:
            if abs(x_vec[2]) < 0.99:
                z_vec = np.array([0, 0, 1])
            else:
                z_vec = np.array([1, 0, 0])
            y_vec = np.cross(z_vec, x_vec)
            y_vec = y_vec / np.linalg.norm(y_vec)
            z_vec = np.cross(x_vec, y_vec)
        T_elem = np.zeros((3, 3))
        T_elem[0, :] = x_vec
        T_elem[1, :] = y_vec
        T_elem[2, :] = z_vec
        T = np.zeros((12, 12))
        for i in range(4):
            T[3 * i:3 * i + 3, 3 * i:3 * i + 3] = T_elem
        u_elem_local = T @ u_elem_global
        E = elem['E']
        A = elem['A']
        N = E * A / L * (u_elem_local[6] - u_elem_local[0])
        I_rho = elem['I_rho']
        kg_local = np.zeros((12, 12))
        coeff = N / L
        kg_local[1, 1] = kg_local[7, 7] = 6 / 5 * coeff
        kg_local[1, 7] = kg_local[7, 1] = -6 / 5 * coeff
        kg_local[1, 5] = kg_local[5, 1] = coeff * L / 10
        kg_local[1, 11] = kg_local[11, 1] = coeff * L / 10
        kg_local[7, 5] = kg_local[5, 7] = -coeff * L / 10
        kg_local[7, 11] = kg_local[11, 7] = -coeff * L / 10
        kg_local[5, 5] = 2 * coeff * L ** 2 / 15
        kg_local[5, 11] = kg_local[11, 5] = -coeff * L ** 2 / 30
        kg_local[11, 11] = 2 * coeff * L ** 2 / 15
        kg_local[2, 2] = kg_local[8, 8] = 6 / 5 * coeff
        kg_local[2, 8] = kg_local[8, 2] = -6 / 5 * coeff
        kg_local[2, 4] = kg_local[4, 2] = -coeff * L / 10
        kg_local[2, 10] = kg_local[10, 2] = -coeff * L / 10
        kg_local[8, 4] = kg_local[4, 8] = coeff * L / 10
        kg_local[8, 10] = kg_local[10, 8] = coeff * L / 10
        kg_local[4, 4] = 2 * coeff * L ** 2 / 15
        kg_local[4, 10] = kg_local[10, 4] = -coeff * L ** 2 / 30
        kg_local[10, 10] = 2 * coeff * L ** 2 / 15
        if I_rho > 0:
            coeff_t = N * I_rho / (A * L)
            kg_local[3, 3] = kg_local[9, 9] = coeff_t
            kg_local[3, 9] = kg_local[9, 3] = -coeff_t
        kg_global = T.T @ kg_local @ T
        for i in range(12):
            for j in range(12):
                K_g[dofs[i], dofs[j]] += kg_global[i, j]
    K_g_ff = np.zeros((n_free, n_free))
    for (i, dof_i) in enumerate(free_dofs):
        for (j, dof_j) in enumerate(free_dofs):
            K_g_ff[i, j] = K_g[dof_i, dof_j]
    (eigenvalues, eigenvectors) = scipy.linalg.eig(K_ff, -K_g_ff)
    positive_indices = []
    for (i, val) in enumerate(eigenvalues):
        if np.isreal(val) and np.real(val) > 1e-10:
            positive_indices.append(i)
    if not positive_indices:
        raise ValueError('No positive eigenvalues found')
    positive_eigenvalues = np.real(eigenvalues[positive_indices])
    min_idx = np.argmin(positive_eigenvalues)
    elastic_critical_load_factor = positive_eigenvalues[min_idx]
    mode_free = np.real(eigenvectors[:, positive_indices[min_idx]])
    deformed_shape_vector = np.zeros(n_dof)
    for (i, dof) in enumerate(free_dofs):
        deformed_shape_vector[dof] = mode_free[i]
    return (elastic_critical_load_factor, deformed_shape_vector)