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

    def compute_element_stiffness_local(elem):
        E = elem['E']
        nu = elem['nu']
        A = elem['A']
        Iy = elem['Iy']
        Iz = elem['Iz']
        J = elem['J']
        node_i = elem['node_i']
        node_j = elem['node_j']
        xi = node_coords[node_i]
        xj = node_coords[node_j]
        L = np.linalg.norm(xj - xi)
        G = E / (2 * (1 + nu))
        k_local = np.zeros((12, 12))
        k_local[0, 0] = E * A / L
        k_local[0, 6] = -E * A / L
        k_local[6, 0] = -E * A / L
        k_local[6, 6] = E * A / L
        k_local[3, 3] = G * J / L
        k_local[3, 9] = -G * J / L
        k_local[9, 3] = -G * J / L
        k_local[9, 9] = G * J / L
        EIz_L3 = E * Iz / L ** 3
        EIz_L2 = E * Iz / L ** 2
        EIz_L = E * Iz / L
        k_local[1, 1] = 12 * EIz_L3
        k_local[1, 5] = 6 * EIz_L2
        k_local[1, 7] = -12 * EIz_L3
        k_local[1, 11] = 6 * EIz_L2
        k_local[5, 1] = 6 * EIz_L2
        k_local[5, 5] = 4 * EIz_L
        k_local[5, 7] = -6 * EIz_L2
        k_local[5, 11] = 2 * EIz_L
        k_local[7, 1] = -12 * EIz_L3
        k_local[7, 5] = -6 * EIz_L2
        k_local[7, 7] = 12 * EIz_L3
        k_local[7, 11] = -6 * EIz_L2
        k_local[11, 1] = 6 * EIz_L2
        k_local[11, 5] = 2 * EIz_L
        k_local[11, 7] = -6 * EIz_L2
        k_local[11, 11] = 4 * EIz_L
        EIy_L3 = E * Iy / L ** 3
        EIy_L2 = E * Iy / L ** 2
        EIy_L = E * Iy / L
        k_local[2, 2] = 12 * EIy_L3
        k_local[2, 4] = -6 * EIy_L2
        k_local[2, 8] = -12 * EIy_L3
        k_local[2, 10] = -6 * EIy_L2
        k_local[4, 2] = -6 * EIy_L2
        k_local[4, 4] = 4 * EIy_L
        k_local[4, 8] = 6 * EIy_L2
        k_local[4, 10] = 2 * EIy_L
        k_local[8, 2] = -12 * EIy_L3
        k_local[8, 4] = 6 * EIy_L2
        k_local[8, 8] = 12 * EIy_L3
        k_local[8, 10] = 6 * EIy_L2
        k_local[10, 2] = -6 * EIy_L2
        k_local[10, 4] = 2 * EIy_L
        k_local[10, 8] = 6 * EIy_L2
        k_local[10, 10] = 4 * EIy_L
        return k_local

    def compute_transformation_matrix(elem):
        node_i = elem['node_i']
        node_j = elem['node_j']
        xi = node_coords[node_i]
        xj = node_coords[node_j]
        e1 = xj - xi
        L = np.linalg.norm(e1)
        e1 = e1 / L
        if 'local_z' in elem and elem['local_z'] is not None:
            e3_ref = np.array(elem['local_z'])
            e3_ref = e3_ref / np.linalg.norm(e3_ref)
            e2 = np.cross(e3_ref, e1)
            e2_norm = np.linalg.norm(e2)
            if e2_norm < 1e-10:
                raise ValueError('local_z is parallel to element axis')
            e2 = e2 / e2_norm
            e3 = np.cross(e1, e2)
        else:
            if abs(e1[2]) > 0.999:
                ref = np.array([1.0, 0.0, 0.0])
            else:
                ref = np.array([0.0, 0.0, 1.0])
            e2 = np.cross(ref, e1)
            e2 = e2 / np.linalg.norm(e2)
            e3 = np.cross(e1, e2)
        R = np.array([e1, e2, e3]).T
        T = np.zeros((12, 12))
        for i in range(4):
            T[3 * i:3 * i + 3, 3 * i:3 * i + 3] = R
        return T

    def compute_element_geometric_stiffness_local(elem, N):
        I_rho = elem['I_rho']
        A = elem['A']
        node_i = elem['node_i']
        node_j = elem['node_j']
        xi = node_coords[node_i]
        xj = node_coords[node_j]
        L = np.linalg.norm(xj - xi)
        kg_local = np.zeros((12, 12))
        N_L = N / L
        I_rho_AL2 = I_rho / (A * L ** 2)
        kg_local[1, 1] = 6 / 5 * N_L
        kg_local[1, 5] = N_L * L / 10
        kg_local[1, 7] = -6 / 5 * N_L
        kg_local[1, 11] = N_L * L / 10
        kg_local[5, 1] = N_L * L / 10
        kg_local[5, 5] = 2 * N_L * L ** 2 / 15
        kg_local[5, 7] = -N_L * L / 10
        kg_local[5, 11] = -N_L * L ** 2 / 30
        kg_local[7, 1] = -6 / 5 * N_L
        kg_local[7, 5] = -N_L * L / 10
        kg_local[7, 7] = 6 / 5 * N_L
        kg_local[7, 11] = -N_L * L / 10
        kg_local[11, 1] = N_L * L / 10
        kg_local[11, 5] = -N_L * L ** 2 / 30
        kg_local[11, 7] = -N_L * L / 10
        kg_local[11, 11] = 2 * N_L * L ** 2 / 15
        kg_local[2, 2] = 6 / 5 * N_L
        kg_local[2, 4] = -N_L * L / 10
        kg_local[2, 8] = -6 / 5 * N_L
        kg_local[2, 10] = -N_L * L / 10
        kg_local[4, 2] = -N_L * L / 10
        kg_local[4, 4] = 2 * N_L * L ** 2 / 15
        kg_local[4, 8] = N_L * L / 10
        kg_local[4, 10] = -N_L * L ** 2 / 30
        kg_local[8, 2] = -6 / 5 * N_L
        kg_local[8, 4] = N_L * L / 10
        kg_local[8, 8] = 6 / 5 * N_L
        kg_local[8, 10] = N_L * L / 10
        kg_local[10, 2] = -N_L * L / 10
        kg_local[10, 4] = -N_L * L ** 2 / 30
        kg_local[10, 8] = N_L * L / 10
        kg_local[10, 10] = 2 * N_L * L ** 2 / 15
        kg_local[3, 3] = N * I_rho_AL2
        kg_local[3, 9] = -N * I_rho_AL2
        kg_local[9, 3] = -N * I_rho_AL2
        kg_local[9, 9] = N * I_rho_AL2
        return kg_local
    K_global = np.zeros((n_dof, n_dof))
    for elem in elements:
        k_local = compute_element_stiffness_local(elem)
        T = compute_transformation_matrix(elem)
        k_global_elem = T.T @ k_local @ T
        node_i = elem['node_i']
        node_j = elem['node_j']
        dof_i = np.arange(6 * node_i, 6 * node_i + 6)
        dof_j = np.arange(6 * node_j, 6 * node_j + 6)
        dof_elem = np.concatenate([dof_i, dof_j])
        for i in range(12):
            for j in range(12):
                K_global[dof_elem[i], dof_elem[j]] += k_global_elem[i, j]
    P_global = np.zeros(n_dof)
    for (node_idx, loads) in nodal_loads.items():
        dof_start = 6 * node_idx
        P_global[dof_start:dof_start + 6] = loads
    constrained_dofs = set()
    for (node_idx, bc_spec) in boundary_conditions.items():
        dof_start = 6 * node_idx
        if all((isinstance(x, bool) for x in bc_spec)):
            for (i, is_fixed) in enumerate(bc_spec):
                if is_fixed:
                    constrained_dofs.add(dof_start + i)
        else:
            for local_dof_idx in bc_spec:
                constrained_dofs.add(dof_start + local_dof_idx)
    constrained_dofs = sorted(list(constrained_dofs))
    free_dofs = [i for i in range(n_dof) if i not in constrained_dofs]
    K_ff = K_global[np.ix_(free_dofs, free_dofs)]
    P_f = P_global[free_dofs]
    u_f = np.linalg.solve(K_ff, P_f)
    u_global = np.zeros(n_dof)
    u_global[free_dofs] = u_f
    Kg_global = np.zeros((n_dof, n_dof))
    for elem in elements:
        node_i = elem['node_i']
        node_j = elem['node_j']
        dof_i = np.arange(6 * node_i, 6 * node_i + 6)
        dof_j = np.arange(6 * node_j, 6 * node_j + 6)
        dof_elem = np.concatenate([dof_i, dof_j])
        u_elem = u_global[dof_elem]
        T = compute_transformation_matrix(elem)
        u_local = T @ u_elem
        xi = node_coords[node_i]
        xj = node_coords[node_j]
        L = np.linalg.norm(xj - xi)
        E = elem['E']
        A = elem['A']
        N = E * A / L * (u_local[6] - u_local[0])
        kg_local = compute_element_geometric_stiffness_local(elem, N)
        kg_global_elem = T.T @ kg_local @ T
        for i in range(12):
            for j in range(12):
                Kg_global[dof_elem[i], dof_elem[j]] += kg_global_elem[i, j]
    K_ff = K_global[np.ix_(free_dofs, free_dofs)]
    Kg_ff = Kg_global[np.ix_(free_dofs, free_dofs)]
    (eigenvalues, eigenvectors) = scipy.linalg.eig(K_ff, -Kg_ff)
    eigenvalues = eigenvalues.real
    eigenvectors = eigenvectors.real
    positive_indices = np.where(eigenvalues > 1e-10)[0]
    if len(positive_indices) == 0:
        raise ValueError('No positive eigenvalues found')
    min_positive_idx = positive_indices[np.argmin(eigenvalues[positive_indices])]
    elastic_critical_load_factor = eigenvalues[min_positive_idx]
    mode_free = eigenvectors[:, min_positive_idx]
    deformed_shape_vector = np.zeros(n_dof)
    deformed_shape_vector[free_dofs] = mode_free
    return (elastic_critical_load_factor, deformed_shape_vector)