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
    n_dofs_total = 6 * n_nodes
    constrained_dofs = set()
    for (node_idx, bc_spec) in boundary_conditions.items():
        if isinstance(bc_spec[0], bool):
            for (dof_local_idx, is_fixed) in enumerate(bc_spec):
                if is_fixed:
                    constrained_dofs.add(6 * node_idx + dof_local_idx)
        else:
            for dof_local_idx in bc_spec:
                constrained_dofs.add(6 * node_idx + dof_local_idx)
    free_dofs = sorted(set(range(n_dofs_total)) - constrained_dofs)
    n_free = len(free_dofs)
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
        coord_i = node_coords[node_i]
        coord_j = node_coords[node_j]
        L = np.linalg.norm(coord_j - coord_i)
        k_local = local_elastic_stiffness_matrix_3D_beam(E, nu, A, L, Iy, Iz, J)
        T = np.eye(12)
        k_global_elem = T.T @ k_local @ T
        dofs_elem = []
        for node in [node_i, node_j]:
            for dof in range(6):
                dofs_elem.append(6 * node + dof)
        for (i_local, i_global) in enumerate(dofs_elem):
            for (j_local, j_global) in enumerate(dofs_elem):
                K_global[i_global, j_global] += k_global_elem[i_local, j_local]
    P_global = np.zeros(n_dofs_total)
    for (node_idx, load_vec) in nodal_loads.items():
        for (dof_local_idx, load_val) in enumerate(load_vec):
            P_global[6 * node_idx + dof_local_idx] += load_val
    K_free = K_global[np.ix_(free_dofs, free_dofs)]
    P_free = P_global[free_dofs]
    u_free = scipy.linalg.solve(K_free, P_free, assume_a='sym')
    u_global = np.zeros(n_dofs_total)
    u_global[free_dofs] = u_free
    K_g_global = np.zeros((n_dofs_total, n_dofs_total))
    for elem in elements:
        node_i = elem['node_i']
        node_j = elem['node_j']
        A = elem['A']
        I_rho = elem['I_rho']
        coord_i = node_coords[node_i]
        coord_j = node_coords[node_j]
        L = np.linalg.norm(coord_j - coord_i)
        dofs_elem = []
        for node in [node_i, node_j]:
            for dof in range(6):
                dofs_elem.append(6 * node + dof)
        u_elem = u_global[dofs_elem]
        k_local = local_elastic_stiffness_matrix_3D_beam(elem['E'], elem['nu'], elem['A'], L, elem['Iy'], elem['Iz'], elem['J'])
        T = np.eye(12)
        k_global_elem = T.T @ k_local @ T
        f_elem = k_global_elem @ u_elem
        Fx2 = -f_elem[6]
        Mx2 = f_elem[9]
        My1 = f_elem[4]
        Mz1 = f_elem[5]
        My2 = f_elem[10]
        Mz2 = f_elem[11]
        k_g_local = local_geometric_stiffness_matrix_3D_beam(L, A, I_rho, Fx2, Mx2, My1, Mz1, My2, Mz2)
        k_g_global_elem = T.T @ k_g_local @ T
        for (i_local, i_global) in enumerate(dofs_elem):
            for (j_local, j_global) in enumerate(dofs_elem):
                K_g_global[i_global, j_global] += k_g_global_elem[i_local, j_local]
    K_free = K_global[np.ix_(free_dofs, free_dofs)]
    K_g_free = K_g_global[np.ix_(free_dofs, free_dofs)]
    try:
        (eigvals, eigvecs) = scipy.linalg.eig(K_free, -K_g_free)
    except:
        (eigvals, eigvecs) = scipy.linalg.eig(K_free, -K_g_free + 1e-12 * np.eye(n_free))
    real_eigvals = np.real(eigvals)
    real_eigvecs = np.real(eigvecs)
    positive_mask = real_eigvals > 0
    if not np.any(positive_mask):
        raise ValueError('No positive eigenvalues found in buckling analysis')
    positive_eigvals = real_eigvals[positive_mask]
    positive_eigvecs = real_eigvecs[:, positive_mask]
    min_idx = np.argmin(positive_eigvals)
    elastic_critical_load_factor = positive_eigvals[min_idx]
    buckling_mode_free = positive_eigvecs[:, min_idx]
    deformed_shape_vector = np.zeros(n_dofs_total)
    deformed_shape_vector[free_dofs] = buckling_mode_free
    return (elastic_critical_load_factor, deformed_shape_vector)