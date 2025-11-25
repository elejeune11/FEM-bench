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
    n_dofs_total = 6 * n_nodes
    K_global = np.zeros((n_dofs_total, n_dofs_total))
    for elem in elements:
        node_i = elem['node_i']
        node_j = elem['node_j']
        E = elem['E']
        nu = elem['nu']
        A = elem['A']
        Iy = elem['Iy']
        Iz = elem['Iz']
        J = elem['J']
        I_rho = elem['I_rho']
        local_z = elem.get('local_z', None)
        coords_i = node_coords[node_i]
        coords_j = node_coords[node_j]
        L_vec = coords_j - coords_i
        L = np.linalg.norm(L_vec)
        k_local = np.zeros((12, 12))
        EA_L = E * A / L
        k_local[0, 0] = EA_L
        k_local[0, 6] = -EA_L
        k_local[6, 0] = -EA_L
        k_local[6, 6] = EA_L
        GJ_L = E / (2 * (1 + nu)) * J / L
        k_local[3, 3] = GJ_L
        k_local[3, 9] = -GJ_L
        k_local[9, 3] = -GJ_L
        k_local[9, 9] = GJ_L
        EIy_L3 = E * Iy / L ** 3
        k_local[2, 2] = 12 * EIy_L3
        k_local[2, 4] = 6 * L * EIy_L3
        k_local[2, 8] = -12 * EIy_L3
        k_local[2, 10] = 6 * L * EIy_L3
        k_local[4, 2] = 6 * L * EIy_L3
        k_local[4, 4] = 4 * L ** 2 * EIy_L3
        k_local[4, 8] = -6 * L * EIy_L3
        k_local[4, 10] = 2 * L ** 2 * EIy_L3
        k_local[8, 2] = -12 * EIy_L3
        k_local[8, 4] = -6 * L * EIy_L3
        k_local[8, 8] = 12 * EIy_L3
        k_local[8, 10] = -6 * L * EIy_L3
        k_local[10, 2] = 6 * L * EIy_L3
        k_local[10, 4] = 2 * L ** 2 * EIy_L3
        k_local[10, 8] = -6 * L * EIy_L3
        k_local[10, 10] = 4 * L ** 2 * EIy_L3
        EIz_L3 = E * Iz / L ** 3
        k_local[1, 1] = 12 * EIz_L3
        k_local[1, 5] = -6 * L * EIz_L3
        k_local[1, 7] = -12 * EIz_L3
        k_local[1, 11] = -6 * L * EIz_L3
        k_local[5, 1] = -6 * L * EIz_L3
        k_local[5, 5] = 4 * L ** 2 * EIz_L3
        k_local[5, 7] = 6 * L * EIz_L3
        k_local[5, 11] = 2 * L ** 2 * EIz_L3
        k_local[7, 1] = -12 * EIz_L3
        k_local[7, 5] = 6 * L * EIz_L3
        k_local[7, 7] = 12 * EIz_L3
        k_local[7, 11] = 6 * L * EIz_L3
        k_local[11, 1] = -6 * L * EIz_L3
        k_local[11, 5] = 2 * L ** 2 * EIz_L3
        k_local[11, 7] = 6 * L * EIz_L3
        k_local[11, 11] = 4 * L ** 2 * EIz_L3
        if local_z is None:
            if abs(L_vec[2]) < 1e-12:
                local_z = [0, 0, 1]
            else:
                local_z = [1, 0, 0]
        local_z = np.array(local_z, dtype=float)
        local_z = local_z / np.linalg.norm(local_z)
        local_x = L_vec / L
        if abs(np.dot(local_x, local_z)) > 1e-06:
            local_z = local_z - np.dot(local_z, local_x) * local_x
            local_z = local_z / np.linalg.norm(local_z)
        local_y = np.cross(local_z, local_x)
        local_y = local_y / np.linalg.norm(local_y)
        local_z = np.cross(local_x, local_y)
        R = np.column_stack([local_x, local_y, local_z])
        T = np.zeros((12, 12))
        T[0:3, 0:3] = R
        T[3:6, 3:6] = R
        T[6:9, 6:9] = R
        T[9:12, 9:12] = R
        k_global_elem = T.T @ k_local @ T
        dofs_i = [node_i * 6 + i for i in range(6)]
        dofs_j = [node_j * 6 + i for i in range(6)]
        all_dofs = dofs_i + dofs_j
        for (i, dof_i) in enumerate(all_dofs):
            for (j, dof_j) in enumerate(all_dofs):
                K_global[dof_i, dof_j] += k_global_elem[i, j]
    P_global = np.zeros(n_dofs_total)
    for (node_idx, load_vec) in nodal_loads.items():
        start_dof = node_idx * 6
        P_global[start_dof:start_dof + 6] = load_vec
    constrained_dofs = set()
    for (node_idx, bc_spec) in boundary_conditions.items():
        if isinstance(bc_spec[0], bool):
            for (dof_local, is_fixed) in enumerate(bc_spec):
                if is_fixed:
                    constrained_dofs.add(node_idx * 6 + dof_local)
        else:
            for dof_local in bc_spec:
                constrained_dofs.add(node_idx * 6 + dof_local)
    free_dofs = [i for i in range(n_dofs_total) if i not in constrained_dofs]
    constrained_dofs_list = sorted(constrained_dofs)
    K_ff = K_global[np.ix_(free_dofs, free_dofs)]
    P_f = P_global[free_dofs]
    try:
        u_f = scipy.linalg.solve(K_ff, P_f, assume_a='sym')
    except scipy.linalg.LinAlgError:
        raise ValueError('Singular stiffness matrix after applying boundary conditions')
    u_full = np.zeros(n_dofs_total)
    u_full[free_dofs] = u_f
    K_g_global = np.zeros((n_dofs_total, n_dofs_total))
    for elem in elements:
        node_i = elem['node_i']
        node_j = elem['node_j']
        E = elem['E']
        A = elem['A']
        I_rho = elem['I_rho']
        local_z = elem.get('local_z', None)
        coords_i = node_coords[node_i]
        coords_j = node_coords[node_j]
        L_vec = coords_j - coords_i
        L = np.linalg.norm(L_vec)
        dofs_i = [node_i * 6 + i for i in range(6)]
        dofs_j = [node_j * 6 + i for i in range(6)]
        u_elem_global = np.concatenate([u_full[dofs_i], u_full[dofs_j]])
        if local_z is None:
            if abs(L_vec[2]) < 1e-12:
                local_z = [0, 0, 1]
            else:
                local_z = [1, 0, 0]
        local_z = np.array(local_z, dtype=float)
        local_z = local_z / np.linalg.norm(local_z)
        local_x = L_vec / L
        if abs(np.dot(local_x, local_z)) > 1e-06:
            local_z = local_z - np.dot(local_z, local_x) * local_x
            local_z = local_z / np.linalg.norm(local_z)
        local_y = np.cross(local_z, local_x)
        local_y = local_y / np.linalg.norm(local_y)
        local_z = np.cross(local_x, local_y)
        R = np.column_stack([local_x, local_y, local_z])
        T = np.zeros((12, 12))
        T[0:3, 0:3] = R
        T[3:6, 3:6] = R
        T[6:9, 6:9] = R
        T[9:12, 9:12] = R
        u_elem_local = T @ u_elem_global
        axial_strain = (u_elem_local[6] - u_elem_local[0]) / L
        axial_force = E * A * axial_strain
        k_g_local = np.zeros((12, 12))
        F = axial_force / L
        k_g_local[0, 0] = F
        k_g_local[0, 6] = -F
        k_g_local[6, 0] = -F
        k_g_local[6, 6] = F
        k_g_local[1, 1] = 6 * F / 5
        k_g_local[1, 5] = F * L / 10
        k_g_local[1, 7] = -6 * F / 5
        k_g_local[1, 11] = F * L / 10
        k_g_local[5, 1] = F * L / 10
        k_g_local[5, 5] = 2 * F * L ** 2 / 15
        k_g_local[5, 7] = -F * L / 10
        k_g_local[5, 11] = -F * L ** 2 / 30
        k_g_local[7, 1] = -6 * F / 5
        k_g_local[7, 5] = -F * L / 10
        k_g_local[7, 7] = 6 * F / 5
        k_g_local[7, 11] = -F * L / 10
        k_g_local[11, 1] = F * L / 10
        k_g_local[11, 5] = -F * L ** 2 / 30
        k_g_local[11, 7] = -F * L / 10
        k_g_local[11, 11] = 2 * F * L ** 2 / 15
        k_g_local[2, 2] = 6 * F / 5
        k_g_local[2, 4] = -F * L / 10
        k_g_local[2, 8] = -6 * F / 5
        k_g_local[2, 10] = -F * L / 10
        k_g_local[4, 2] = -F * L / 10
        k_g_local[4, 4] = 2 * F * L ** 2 / 15
        k_g_local[4, 8] = F * L / 10
        k_g_local[4, 10] = -F * L ** 2 / 30
        k_g_local[8, 2] = -6 * F / 5
        k_g_local[8, 4] = F * L / 10
        k_g_local[8, 8] = 6 * F / 5
        k_g_local[8, 10] = F * L / 10
        k_g_local[10, 2] = -F * L / 10
        k_g_local[10, 4] = -F * L ** 2 / 30
        k_g_local[10, 8] = F * L / 10
        k_g_local[10, 10] = 2 * F * L ** 2 / 15
        k_g_global_elem = T.T @ k_g_local @ T
        all_dofs = dofs_i + dofs_j
        for (i, dof_i) in enumerate(all_dofs):
            for (j, dof_j) in enumerate(all_dofs):
                K_g_global[dof_i, dof_j] += k_g_global_elem[i, j]
    K_ff = K_global[np.ix_(free_dofs, free_dofs)]
    K_g_ff = K_g_global[np.ix_(free_dofs, free_dofs)]
    try:
        (eigenvalues, eigenvectors) = scipy.linalg.eigh(K_ff, -K_g_ff)
    except (scipy.linalg.LinAlgError, ValueError):
        (eigenvalues, eigenvectors) = scipy.linalg.eig(K_ff, -K_g_ff)
        eigenvalues = eigenvalues.real
        eigenvectors = eigenvectors.real
    positive_eigenvalues = eigenvalues[eigenvalues > 0]
    if len(positive_eigenvalues) == 0:
        raise ValueError('No positive eigenvalues found - structure may be unstable or loading inappropriate')
    min_positive_idx = np.argmin(positive_eigenvalues)
    elastic_critical_load_factor = positive_eigenvalues[min_positive_idx]
    original_idx = np.where(eigenvalues == elastic_critical_load_factor)[0][0]
    mode_free = eigenvectors[:, original_idx]
    deformed_shape_vector = np.zeros(n_dofs_total)
    deformed_shape_vector[free_dofs] = mode_free
    return (elastic_critical_load_factor, deformed_shape_vector)