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
        Mapping from node index -> length-6 vector of load components applied at
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
    for element in elements:
        node_i = element['node_i']
        node_j = element['node_j']
        coords_i = node_coords[node_i]
        coords_j = node_coords[node_j]
        E = element['E']
        nu = element['nu']
        A = element['A']
        I_y = element['I_y']
        I_z = element['I_z']
        J = element['J']
        I_rho = element['I_rho']
        local_z = element.get('local_z', None)
        L_vec = coords_j - coords_i
        L = np.linalg.norm(L_vec)
        K_local = np.zeros((12, 12))
        EA_L = E * A / L
        K_local[0, 0] = EA_L
        K_local[0, 6] = -EA_L
        K_local[6, 0] = -EA_L
        K_local[6, 6] = EA_L
        G = E / (2 * (1 + nu))
        GJ_L = G * J / L
        K_local[3, 3] = GJ_L
        K_local[3, 9] = -GJ_L
        K_local[9, 3] = -GJ_L
        K_local[9, 9] = GJ_L
        EIy_L3 = E * I_y / L ** 3
        K_local[1, 1] = 12 * EIy_L3
        K_local[1, 5] = 6 * L * EIy_L3
        K_local[1, 7] = -12 * EIy_L3
        K_local[1, 11] = 6 * L * EIy_L3
        K_local[5, 1] = 6 * L * EIy_L3
        K_local[5, 5] = 4 * L ** 2 * EIy_L3
        K_local[5, 7] = -6 * L * EIy_L3
        K_local[5, 11] = 2 * L ** 2 * EIy_L3
        K_local[7, 1] = -12 * EIy_L3
        K_local[7, 5] = -6 * L * EIy_L3
        K_local[7, 7] = 12 * EIy_L3
        K_local[7, 11] = -6 * L * EIy_L3
        K_local[11, 1] = 6 * L * EIy_L3
        K_local[11, 5] = 2 * L ** 2 * EIy_L3
        K_local[11, 7] = -6 * L * EIy_L3
        K_local[11, 11] = 4 * L ** 2 * EIy_L3
        EIz_L3 = E * I_z / L ** 3
        K_local[2, 2] = 12 * EIz_L3
        K_local[2, 4] = -6 * L * EIz_L3
        K_local[2, 8] = -12 * EIz_L3
        K_local[2, 10] = -6 * L * EIz_L3
        K_local[4, 2] = -6 * L * EIz_L3
        K_local[4, 4] = 4 * L ** 2 * EIz_L3
        K_local[4, 8] = 6 * L * EIz_L3
        K_local[4, 10] = 2 * L ** 2 * EIz_L3
        K_local[8, 2] = -12 * EIz_L3
        K_local[8, 4] = 6 * L * EIz_L3
        K_local[8, 8] = 12 * EIz_L3
        K_local[8, 10] = 6 * L * EIz_L3
        K_local[10, 2] = -6 * L * EIz_L3
        K_local[10, 4] = 2 * L ** 2 * EIz_L3
        K_local[10, 8] = 6 * L * EIz_L3
        K_local[10, 10] = 4 * L ** 2 * EIz_L3
        if local_z is None:
            local_x = L_vec / L
            if abs(local_x[0]) > 1e-10 or abs(local_x[1]) > 1e-10:
                temp_vec = np.array([0, 0, 1])
            else:
                temp_vec = np.array([1, 0, 0])
            local_y = np.cross(local_z, local_x)
            local_y = local_y / np.linalg.norm(local_y)
            local_z = np.cross(local_x, local_y)
            local_z = local_z / np.linalg.norm(local_z)
        else:
            local_z = np.array(local_z)
            local_z = local_z / np.linalg.norm(local_z)
            local_x = L_vec / L
            local_y = np.cross(local_z, local_x)
            local_y = local_y / np.linalg.norm(local_y)
            local_z = np.cross(local_x, local_y)
            local_z = local_z / np.linalg.norm(local_z)
        R = np.zeros((3, 3))
        R[:, 0] = local_x
        R[:, 1] = local_y
        R[:, 2] = local_z
        T = np.zeros((12, 12))
        T[0:3, 0:3] = R
        T[3:6, 3:6] = R
        T[6:9, 6:9] = R
        T[9:12, 9:12] = R
        K_global_elem = T.T @ K_local @ T
        dofs_i = [6 * node_i + i for i in range(6)]
        dofs_j = [6 * node_j + i for i in range(6)]
        all_dofs = dofs_i + dofs_j
        for (idx_i, dof_i) in enumerate(all_dofs):
            for (idx_j, dof_j) in enumerate(all_dofs):
                K_global[dof_i, dof_j] += K_global_elem[idx_i, idx_j]
    P_global = np.zeros(n_dofs_total)
    for (node_idx, load_vec) in nodal_loads.items():
        start_dof = 6 * node_idx
        P_global[start_dof:start_dof + 6] = load_vec
    constrained_dofs = set()
    for (node_idx, bc_spec) in boundary_conditions.items():
        if isinstance(bc_spec[0], bool):
            for (dof_local, is_constrained) in enumerate(bc_spec):
                if is_constrained:
                    constrained_dofs.add(6 * node_idx + dof_local)
        else:
            for dof_local in bc_spec:
                constrained_dofs.add(6 * node_idx + dof_local)
    free_dofs = [i for i in range(n_dofs_total) if i not in constrained_dofs]
    n_free = len(free_dofs)
    K_ff = K_global[np.ix_(free_dofs, free_dofs)]
    P_f = P_global[free_dofs]
    try:
        u_f = scipy.linalg.solve(K_ff, P_f, assume_a='sym')
    except scipy.linalg.LinAlgError:
        raise ValueError('Singular stiffness matrix after applying boundary conditions')
    u_full = np.zeros(n_dofs_total)
    u_full[free_dofs] = u_f
    K_g_global = np.zeros((n_dofs_total, n_dofs_total))
    for element in elements:
        node_i = element['node_i']
        node_j = element['node_j']
        coords_i = node_coords[node_i]
        coords_j = node_coords[node_j]
        A = element['A']
        I_rho = element['I_rho']
        local_z = element.get('local_z', None)
        L_vec = coords_j - coords_i
        L = np.linalg.norm(L_vec)
        dofs_i = [6 * node_i + i for i in range(6)]
        dofs_j = [6 * node_j + i for i in range(6)]
        u_elem = np.concatenate([u_full[dofs_i], u_full[dofs_j]])
        if local_z is None:
            local_x = L_vec / L
            if abs(local_x[0]) > 1e-10 or abs(local_x[1]) > 1e-10:
                temp_vec = np.array([0, 0, 1])
            else:
                temp_vec = np.array([1, 0, 0])
            local_y = np.cross(temp_vec, local_x)
            local_y = local_y / np.linalg.norm(local_y)
            local_z = np.cross(local_x, local_y)
            local_z = local_z / np.linalg.norm(local_z)
        else:
            local_z = np.array(local_z)
            local_z = local_z / np.linalg.norm(local_z)
            local_x = L_vec / L
            local_y = np.cross(local_z, local_x)
            local_y = local_y / np.linalg.norm(local_y)
            local_z = np.cross(local_x, local_y)
            local_z = local_z / np.linalg.norm(local_z)
        R = np.zeros((3, 3))
        R[:, 0] = local_x
        R[:, 1] = local_y
        R[:, 2] = local_z
        T = np.zeros((12, 12))
        T[0:3, 0:3] = R
        T[3:6, 3:6] = R
        T[6:9, 6:9] = R
        T[9:12, 9:12] = R
        u_elem_local = T @ u_elem
        N = E * A / L * (u_elem_local[6] - u_elem_local[0])
        K_g_local = np.zeros((12, 12))
        N_L = N / L
        K_g_local[0, 0] = N_L
        K_g_local[0, 6] = -N_L
        K_g_local[6, 0] = -N_L
        K_g_local[6, 6] = N_L
        K_g_local[1, 1] = 6 * N_L / 5
        K_g_local[1, 5] = N_L / 10
        K_g_local[1, 7] = -6 * N_L / 5
        K_g_local[1, 11] = N_L / 10
        K_g_local[5, 1] = N_L / 10
        K_g_local[5, 5] = 2 * L * N_L / 15
        K_g_local[5, 7] = -N_L / 10
        K_g_local[5, 11] = -L * N_L / 30
        K_g_local[7, 1] = -6 * N_L / 5
        K_g_local[7, 5] = -N_L / 10
        K_g_local[7, 7] = 6 * N_L / 5
        K_g_local[7, 11] = -N_L / 10
        K_g_local[11, 1] = N_L / 10
        K_g_local[11, 5] = -L * N_L / 30
        K_g_local[11, 7] = -N_L / 10
        K_g_local[11, 11] = 2 * L * N_L / 15
        K_g_local[2, 2] = 6 * N_L / 5
        K_g_local[2, 4] = -N_L / 10
        K_g_local[2, 8] = -6 * N_L / 5
        K_g_local[2, 10] = -N_L / 10
        K_g_local[4, 2] = -N_L / 10
        K_g_local[4, 4] = 2 * L * N_L / 15
        K_g_local[4, 8] = N_L / 10
        K_g_local[4, 10] = -L * N_L / 30
        K_g_local[8, 2] = -6 * N_L / 5
        K_g_local[8, 4] = N_L / 10
        K_g_local[8, 8] = 6 * N_L / 5
        K_g_local[8, 10] = N_L / 10
        K_g_local[10, 2] = -N_L / 10
        K_g_local[10, 4] = -L * N_L / 30
        K_g_local[10, 8] = N_L / 10
        K_g_local[10, 10] = 2 * L * N_L / 15
        K_g_global_elem = T.T @ K_g_local @ T
        for (idx_i, dof_i) in enumerate(all_dofs):
            for (idx_j, dof_j) in enumerate(all_dofs):
                K_g_global[dof_i, dof_j] += K_g_global_elem[idx_i, idx_j]
    K_ff = K_global[np.ix_(free_dofs, free_dofs)]
    K_g_ff = K_g_global[np.ix_(free_dofs, free_dofs)]
    try:
        (eigenvalues, eigenvectors) = scipy.linalg.eigh(K_ff, -K_g_ff)
    except (scipy.linalg.LinAlgError, ValueError):
        raise ValueError('Generalized eigenproblem solution failed')
    positive_eigenvalues = eigenvalues[eigenvalues > 0]
    if len(positive_eigenvalues) == 0:
        raise ValueError('No positive eigenvalues found')
    min_positive_idx = np.argmin(positive_eigenvalues)
    elastic_critical_load_factor = positive_eigenvalues[min_positive_idx]
    original_eigenvalue_idx = np.where(eigenvalues == elastic_critical_load_factor)[0][0]
    buckling_mode_free = eigenvectors[:, original_eigenvalue_idx]
    deformed_shape_vector = np.zeros(n_dofs_total)
    deformed_shape_vector[free_dofs] = buckling_mode_free
    return (elastic_critical_load_factor, deformed_shape_vector)