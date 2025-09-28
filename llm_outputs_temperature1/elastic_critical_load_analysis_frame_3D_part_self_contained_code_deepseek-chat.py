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
        (xi, yi, zi) = node_coords[node_i]
        (xj, yj, zj) = node_coords[node_j]
        L = np.sqrt((xj - xi) ** 2 + (yj - yi) ** 2 + (zj - zi) ** 2)
        E = elem['E']
        nu = elem['nu']
        A = elem['A']
        Iy = elem['Iy']
        Iz = elem['Iz']
        J = elem['J']
        k_local = local_elastic_stiffness_matrix_3D_beam(E, nu, A, L, Iy, Iz, J)
        T = np.eye(12)
        k_global_elem = T.T @ k_local @ T
        dof_i = [6 * node_i + i for i in range(6)]
        dof_j = [6 * node_j + i for i in range(6)]
        dof_indices = dof_i + dof_j
        for (i, idx_i) in enumerate(dof_indices):
            for (j, idx_j) in enumerate(dof_indices):
                K_global[idx_i, idx_j] += k_global_elem[i, j]
    P_global = np.zeros(n_dof_total)
    for (node_idx, load_vec) in nodal_loads.items():
        dof_start = 6 * node_idx
        P_global[dof_start:dof_start + 6] = load_vec
    constrained_dofs = set()
    for (node_idx, bc) in boundary_conditions.items():
        if isinstance(bc[0], bool):
            for (dof_local, is_constrained) in enumerate(bc):
                if is_constrained:
                    constrained_dofs.add(6 * node_idx + dof_local)
        else:
            for dof_local in bc:
                constrained_dofs.add(6 * node_idx + dof_local)
    free_dofs = sorted(set(range(n_dof_total)) - constrained_dofs)
    n_free = len(free_dofs)
    K_free = K_global[np.ix_(free_dofs, free_dofs)]
    P_free = P_global[free_dofs]
    try:
        u_free = scipy.linalg.solve(K_free, P_free, assume_a='sym')
    except scipy.linalg.LinAlgError:
        raise ValueError('Singular stiffness matrix - check boundary conditions')
    u_full = np.zeros(n_dof_total)
    u_full[free_dofs] = u_free
    K_g_global = np.zeros((n_dof_total, n_dof_total))
    for elem in elements:
        node_i = elem['node_i']
        node_j = elem['node_j']
        (xi, yi, zi) = node_coords[node_i]
        (xj, yj, zj) = node_coords[node_j]
        L = np.sqrt((xj - xi) ** 2 + (yj - yi) ** 2 + (zj - zi) ** 2)
        A = elem['A']
        I_rho = elem['I_rho']
        dof_i = [6 * node_i + i for i in range(6)]
        dof_j = [6 * node_j + i for i in range(6)]
        Fx2 = nodal_loads.get(node_j, [0, 0, 0, 0, 0, 0])[0] if node_j in nodal_loads else 0
        Mx2 = nodal_loads.get(node_j, [0, 0, 0, 0, 0, 0])[3] if node_j in nodal_loads else 0
        My1 = nodal_loads.get(node_i, [0, 0, 0, 0, 0, 0])[4] if node_i in nodal_loads else 0
        Mz1 = nodal_loads.get(node_i, [0, 0, 0, 0, 0, 0])[5] if node_i in nodal_loads else 0
        My2 = nodal_loads.get(node_j, [0, 0, 0, 0, 0, 0])[4] if node_j in nodal_loads else 0
        Mz2 = nodal_loads.get(node_j, [0, 0, 0, 0, 0, 0])[5] if node_j in nodal_loads else 0
        k_g_local = local_geometric_stiffness_matrix_3D_beam(L, A, I_rho, Fx2, Mx2, My1, Mz1, My2, Mz2)
        T = np.eye(12)
        k_g_global_elem = T.T @ k_g_local @ T
        dof_indices = dof_i + dof_j
        for (i, idx_i) in enumerate(dof_indices):
            for (j, idx_j) in enumerate(dof_indices):
                K_g_global[idx_i, idx_j] += k_g_global_elem[i, j]
    K_free = K_global[np.ix_(free_dofs, free_dofs)]
    K_g_free = K_g_global[np.ix_(free_dofs, free_dofs)]
    try:
        (eigenvalues, eigenvectors) = scipy.linalg.eigh(K_free, -K_g_free)
    except (scipy.linalg.LinAlgError, ValueError):
        try:
            (eigenvalues, eigenvectors) = scipy.sparse.linalg.eigsh(K_free, k=6, M=-K_g_free, which='SM')
        except:
            (eigenvalues, eigenvectors) = scipy.linalg.eig(K_free, -K_g_free)
    positive_eigenvalues = eigenvalues[eigenvalues > 0]
    if len(positive_eigenvalues) == 0:
        raise ValueError('No positive eigenvalues found - check model and loading')
    min_positive_idx = np.argmin(positive_eigenvalues)
    elastic_critical_load_factor = positive_eigenvalues[min_positive_idx]
    eigenvector_free = eigenvectors[:, np.where(eigenvalues == elastic_critical_load_factor)[0][0]]
    deformed_shape_vector = np.zeros(n_dof_total)
    deformed_shape_vector[free_dofs] = eigenvector_free.real
    return (float(elastic_critical_load_factor), deformed_shape_vector)