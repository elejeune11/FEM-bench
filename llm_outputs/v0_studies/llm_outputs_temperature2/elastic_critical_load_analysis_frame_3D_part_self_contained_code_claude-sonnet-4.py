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
    n_dof = 6 * n_nodes
    K = np.zeros((n_dof, n_dof))
    P = np.zeros(n_dof)
    for (node_idx, loads) in nodal_loads.items():
        for (i, load) in enumerate(loads):
            P[6 * node_idx + i] = load

    def compute_transformation_matrix(node_i_coords, node_j_coords, local_z_dir=None):
        dx = node_j_coords - node_i_coords
        L = np.linalg.norm(dx)
        local_x = dx / L
        if local_z_dir is not None:
            local_z_dir = np.array(local_z_dir)
            local_z_dir = local_z_dir / np.linalg.norm(local_z_dir)
            local_z_dir = local_z_dir - np.dot(local_z_dir, local_x) * local_x
            local_z = local_z_dir / np.linalg.norm(local_z_dir)
        else:
            global_z = np.array([0, 0, 1])
            if abs(np.dot(local_x, global_z)) < 0.99:
                local_z_temp = global_z - np.dot(global_z, local_x) * local_x
                local_z = local_z_temp / np.linalg.norm(local_z_temp)
            else:
                global_y = np.array([0, 1, 0])
                local_z_temp = global_y - np.dot(global_y, local_x) * local_x
                local_z = local_z_temp / np.linalg.norm(local_z_temp)
        local_y = np.cross(local_z, local_x)
        R = np.array([local_x, local_y, local_z]).T
        T = np.zeros((12, 12))
        for i in range(4):
            T[3 * i:3 * i + 3, 3 * i:3 * i + 3] = R
        return (T, L)
    for element in elements:
        node_i = element['node_i']
        node_j = element['node_j']
        E = element['E']
        nu = element['nu']
        A = element['A']
        Iy = element['Iy']
        Iz = element['Iz']
        J = element['J']
        local_z_dir = element.get('local_z', None)
        node_i_coords = node_coords[node_i]
        node_j_coords = node_coords[node_j]
        (T, L) = compute_transformation_matrix(node_i_coords, node_j_coords, local_z_dir)
        k_local = local_elastic_stiffness_matrix_3D_beam(E, nu, A, L, Iy, Iz, J)
        k_global = T.T @ k_local @ T
        dofs_i = np.arange(6 * node_i, 6 * node_i + 6)
        dofs_j = np.arange(6 * node_j, 6 * node_j + 6)
        dofs = np.concatenate([dofs_i, dofs_j])
        for i in range(12):
            for j in range(12):
                K[dofs[i], dofs[j]] += k_global[i, j]
    constrained_dofs = set()
    for (node_idx, bc_spec) in boundary_conditions.items():
        if len(bc_spec) == 6 and all((isinstance(x, (bool, np.bool_)) for x in bc_spec)):
            for (i, is_constrained) in enumerate(bc_spec):
                if is_constrained:
                    constrained_dofs.add(6 * node_idx + i)
        else:
            for dof_idx in bc_spec:
                constrained_dofs.add(6 * node_idx + dof_idx)
    all_dofs = set(range(n_dof))
    free_dofs = sorted(all_dofs - constrained_dofs)
    if len(free_dofs) == 0:
        raise ValueError('No free DOFs available')
    K_free = K[np.ix_(free_dofs, free_dofs)]
    P_free = P[free_dofs]
    try:
        u_free = np.linalg.solve(K_free, P_free)
    except np.linalg.LinAlgError:
        raise ValueError('Singular stiffness matrix')
    u_full = np.zeros(n_dof)
    u_full[free_dofs] = u_free
    K_g = np.zeros((n_dof, n_dof))
    for element in elements:
        node_i = element['node_i']
        node_j = element['node_j']
        A = element['A']
        I_rho = element['I_rho']
        local_z_dir = element.get('local_z', None)
        node_i_coords = node_coords[node_i]
        node_j_coords = node_coords[node_j]
        (T, L) = compute_transformation_matrix(node_i_coords, node_j_coords, local_z_dir)
        dofs_i = np.arange(6 * node_i, 6 * node_i + 6)
        dofs_j = np.arange(6 * node_j, 6 * node_j + 6)
        dofs = np.concatenate([dofs_i, dofs_j])
        u_element = u_full[dofs]
        u_local = T @ u_element
        k_local = local_elastic_stiffness_matrix_3D_beam(element['E'], element['nu'], A, L, element['Iy'], element['Iz'], element['J'])
        f_local = k_local @ u_local
        Fx2 = -f_local[0]
        Mx2 = f_local[3]
        My1 = f_local[4]
        Mz1 = f_local[5]
        My2 = f_local[10]
        Mz2 = f_local[11]
        k_g_local = local_geometric_stiffness_matrix_3D_beam(L, A, I_rho, Fx2, Mx2, My1, Mz1, My2, Mz2)
        k_g_global = T.T @ k_g_local @ T
        for i in range(12):
            for j in range(12):
                K_g[dofs[i], dofs[j]] += k_g_global[i, j]
    K_free = K[np.ix_(free_dofs, free_dofs)]
    K_g_free = K_g[np.ix_(free_dofs, free_dofs)]
    try:
        (eigenvalues, eigenvectors) = scipy.linalg.eigh(K_free, -K_g_free)
    except np.linalg.LinAlgError:
        raise ValueError('Failed to solve eigenvalue problem')
    positive_eigenvalues = eigenvalues[eigenvalues > 1e-10]
    if len(positive_eigenvalues) == 0:
        raise ValueError('No positive eigenvalues found')
    min_eigenvalue = np.min(positive_eigenvalues)
    min_index = np.where(eigenvalues == min_eigenvalue)[0][0]
    mode_free = eigenvectors[:, min_index]
    mode_full = np.zeros(n_dof)
    mode_full[free_dofs] = mode_free
    return (min_eigenvalue, mode_full)