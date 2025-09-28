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
    n_nodes = len(node_coords)
    n_dof = 6 * n_nodes
    K = np.zeros((n_dof, n_dof))
    P = np.zeros(n_dof)

    def build_transformation_matrix(node_i, node_j, local_z):
        xi = node_coords[node_i]
        xj = node_coords[node_j]
        x_local = xj - xi
        L = np.linalg.norm(x_local)
        x_local = x_local / L
        if local_z is None:
            if abs(x_local[2]) < 0.999:
                z_local = np.array([0, 0, 1])
            else:
                z_local = np.array([0, 1, 0])
        else:
            z_local = np.array(local_z)
        y_local = np.cross(z_local, x_local)
        y_local = y_local / np.linalg.norm(y_local)
        z_local = np.cross(x_local, y_local)
        R = np.array([x_local, y_local, z_local]).T
        T = np.zeros((12, 12))
        T[0:3, 0:3] = R
        T[3:6, 3:6] = R
        T[6:9, 6:9] = R
        T[9:12, 9:12] = R
        return (T, L)
    for elem in elements:
        node_i = elem['node_i']
        node_j = elem['node_j']
        local_z = elem.get('local_z', None)
        (T, L) = build_transformation_matrix(node_i, node_j, local_z)
        k_local = local_elastic_stiffness_matrix_3D_beam(elem['E'], elem['nu'], elem['A'], L, elem['Iy'], elem['Iz'], elem['J'])
        k_global = T.T @ k_local @ T
        dof_i = list(range(6 * node_i, 6 * node_i + 6))
        dof_j = list(range(6 * node_j, 6 * node_j + 6))
        dof_elem = dof_i + dof_j
        for (i, gi) in enumerate(dof_elem):
            for (j, gj) in enumerate(dof_elem):
                K[gi, gj] += k_global[i, j]
    for (node, loads) in nodal_loads.items():
        for (i, load) in enumerate(loads):
            P[6 * node + i] = load
    constrained_dofs = set()
    for (node, bcs) in boundary_conditions.items():
        if all((isinstance(bc, bool) for bc in bcs)):
            for (i, is_fixed) in enumerate(bcs):
                if is_fixed:
                    constrained_dofs.add(6 * node + i)
        else:
            for local_dof in bcs:
                constrained_dofs.add(6 * node + local_dof)
    free_dofs = [i for i in range(n_dof) if i not in constrained_dofs]
    n_free = len(free_dofs)
    K_ff = K[np.ix_(free_dofs, free_dofs)]
    P_f = P[free_dofs]
    u_f = np.linalg.solve(K_ff, P_f)
    u = np.zeros(n_dof)
    u[free_dofs] = u_f
    K_g = np.zeros((n_dof, n_dof))
    for elem in elements:
        node_i = elem['node_i']
        node_j = elem['node_j']
        local_z = elem.get('local_z', None)
        (T, L) = build_transformation_matrix(node_i, node_j, local_z)
        dof_i = list(range(6 * node_i, 6 * node_i + 6))
        dof_j = list(range(6 * node_j, 6 * node_j + 6))
        dof_elem = dof_i + dof_j
        u_elem_global = u[dof_elem]
        u_elem_local = T @ u_elem_global
        k_local = local_elastic_stiffness_matrix_3D_beam(elem['E'], elem['nu'], elem['A'], L, elem['Iy'], elem['Iz'], elem['J'])
        f_local = k_local @ u_elem_local
        Fx2 = -f_local[6]
        Mx2 = -f_local[9]
        My1 = f_local[4]
        Mz1 = f_local[5]
        My2 = -f_local[10]
        Mz2 = -f_local[11]
        kg_local = local_geometric_stiffness_matrix_3D_beam(L, elem['A'], elem['I_rho'], Fx2, Mx2, My1, Mz1, My2, Mz2)
        kg_global = T.T @ kg_local @ T
        for (i, gi) in enumerate(dof_elem):
            for (j, gj) in enumerate(dof_elem):
                K_g[gi, gj] += kg_global[i, j]
    K_g_ff = K_g[np.ix_(free_dofs, free_dofs)]
    (eigvals, eigvecs) = scipy.linalg.eig(K_ff, -K_g_ff)
    real_eigvals = np.real(eigvals)
    positive_mask = real_eigvals > 1e-10
    if not np.any(positive_mask):
        raise ValueError('No positive eigenvalues found')
    positive_eigvals = real_eigvals[positive_mask]
    min_idx_in_positive = np.argmin(positive_eigvals)
    min_positive_eigval = positive_eigvals[min_idx_in_positive]
    positive_indices = np.where(positive_mask)[0]
    min_idx = positive_indices[min_idx_in_positive]
    mode_free = np.real(eigvecs[:, min_idx])
    mode_full = np.zeros(n_dof)
    mode_full[free_dofs] = mode_free
    return (min_positive_eigval, mode_full)