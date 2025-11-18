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
        All constrained DOFs are removed from the free set. It is the caller’s
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
    n_dof_total = 6 * n_nodes
    K_global = np.zeros((n_dof_total, n_dof_total))
    for elem in elements:
        node_i = elem['node_i']
        node_j = elem['node_j']
        coords_i = node_coords[node_i]
        coords_j = node_coords[node_j]
        L = np.linalg.norm(coords_j - coords_i)
        E = elem['E']
        nu = elem['nu']
        A = elem['A']
        I_y = elem['I_y']
        I_z = elem['I_z']
        J = elem['J']
        k_local = local_elastic_stiffness_matrix_3D_beam(E, nu, A, L, I_y, I_z, J)
        if 'local_z' in elem and elem['local_z'] is not None:
            local_z = np.array(elem['local_z'])
            local_z = local_z / np.linalg.norm(local_z)
            local_x = (coords_j - coords_i) / L
            local_y = np.cross(local_z, local_x)
            local_y = local_y / np.linalg.norm(local_y)
            local_z = np.cross(local_x, local_y)
        else:
            local_x = (coords_j - coords_i) / L
            if abs(local_x[2]) > 0.9:
                local_y = np.array([0.0, 1.0, 0.0])
            else:
                local_y = np.array([0.0, 0.0, 1.0])
            local_z = np.cross(local_x, local_y)
            local_z = local_z / np.linalg.norm(local_z)
            local_y = np.cross(local_z, local_x)
        R = np.zeros((3, 3))
        R[:, 0] = local_x
        R[:, 1] = local_y
        R[:, 2] = local_z
        T = np.zeros((12, 12))
        T[0:3, 0:3] = R
        T[3:6, 3:6] = R
        T[6:9, 6:9] = R
        T[9:12, 9:12] = R
        k_global_elem = T.T @ k_local @ T
        dofs_i = [6 * node_i + i for i in range(6)]
        dofs_j = [6 * node_j + i for i in range(6)]
        dofs = dofs_i + dofs_j
        for (i, dof_i) in enumerate(dofs):
            for (j, dof_j) in enumerate(dofs):
                K_global[dof_i, dof_j] += k_global_elem[i, j]
    P_global = np.zeros(n_dof_total)
    for (node_idx, load_vec) in nodal_loads.items():
        dof_start = 6 * node_idx
        P_global[dof_start:dof_start + 6] = load_vec
    constrained_dofs = set()
    for (node_idx, bc_spec) in boundary_conditions.items():
        if isinstance(bc_spec[0], bool):
            for (dof_local, is_constrained) in enumerate(bc_spec):
                if is_constrained:
                    constrained_dofs.add(6 * node_idx + dof_local)
        else:
            for dof_local in bc_spec:
                constrained_dofs.add(6 * node_idx + dof_local)
    free_dofs = [i for i in range(n_dof_total) if i not in constrained_dofs]
    n_free = len(free_dofs)
    K_free = K_global[np.ix_(free_dofs, free_dofs)]
    P_free = P_global[free_dofs]
    u_free = scipy.linalg.solve(K_free, P_free, assume_a='sym')
    u_full = np.zeros(n_dof_total)
    u_full[free_dofs] = u_free
    K_g_global = np.zeros((n_dof_total, n_dof_total))
    for elem in elements:
        node_i = elem['node_i']
        node_j = elem['node_j']
        coords_i = node_coords[node_i]
        coords_j = node_coords[node_j]
        L = np.linalg.norm(coords_j - coords_i)
        A = elem['A']
        I_rho = elem['I_rho']
        if 'local_z' in elem and elem['local_z'] is not None:
            local_z = np.array(elem['local_z'])
            local_z = local_z / np.linalg.norm(local_z)
            local_x = (coords_j - coords_i) / L
            local_y = np.cross(local_z, local_x)
            local_y = local_y / np.linalg.norm(local_y)
            local_z = np.cross(local_x, local_y)
        else:
            local_x = (coords_j - coords_i) / L
            if abs(local_x[2]) > 0.9:
                local_y = np.array([0.0, 1.0, 0.0])
            else:
                local_y = np.array([0.0, 0.0, 1.0])
            local_z = np.cross(local_x, local_y)
            local_z = local_z / np.linalg.norm(local_z)
            local_y = np.cross(local_z, local_x)
        R = np.zeros((3, 3))
        R[:, 0] = local_x
        R[:, 1] = local_y
        R[:, 2] = local_z
        T = np.zeros((12, 12))
        T[0:3, 0:3] = R
        T[3:6, 3:6] = R
        T[6:9, 6:9] = R
        T[9:12, 9:12] = R
        dofs_i = [6 * node_i + i for i in range(6)]
        dofs_j = [6 * node_j + i for i in range(6)]
        u_elem_global = np.concatenate([u_full[dofs_i], u_full[dofs_j]])
        u_elem_local = T @ u_elem_global
        k_local = local_elastic_stiffness_matrix_3D_beam(elem['E'], elem['nu'], elem['A'], L, elem['I_y'], elem['I_z'], elem['J'])
        f_elem_local = k_local @ u_elem_local
        Fx2 = -f_elem_local[6]
        Mx2 = f_elem_local[9]
        My1 = f_elem_local[4]
        Mz1 = f_elem_local[5]
        My2 = f_elem_local[10]
        Mz2 = f_elem_local[11]
        k_g_local = local_geometric_stiffness_matrix_3D_beam(L, A, I_rho, Fx2, Mx2, My1, Mz1, My2, Mz2)
        k_g_global_elem = T.T @ k_g_local @ T
        dofs = dofs_i + dofs_j
        for (i, dof_i) in enumerate(dofs):
            for (j, dof_j) in enumerate(dofs):
                K_g_global[dof_i, dof_j] += k_g_global_elem[i, j]
    K_g_free = K_g_global[np.ix_(free_dofs, free_dofs)]
    try:
        (eigenvalues, eigenvectors) = scipy.linalg.eigh(K_free, -K_g_free)
    except (scipy.linalg.LinAlgError, ValueError):
        (eigenvalues, eigenvectors) = scipy.linalg.eig(K_free, -K_g_free)
        eigenvalues = eigenvalues.real
        eigenvectors = eigenvectors.real
    positive_eigenvalues = eigenvalues[eigenvalues > 0]
    if len(positive_eigenvalues) == 0:
        raise ValueError('No positive eigenvalues found in buckling analysis')
    min_positive_idx = np.argmin(positive_eigenvalues)
    elastic_critical_load_factor = positive_eigenvalues[min_positive_idx]
    eigenvector_free = eigenvectors[:, np.where(eigenvalues == elastic_critical_load_factor)[0][0]]
    deformed_shape_vector = np.zeros(n_dof_total)
    deformed_shape_vector[free_dofs] = eigenvector_free
    return (elastic_critical_load_factor, deformed_shape_vector)