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
    n_dof = 6 * n_nodes
    K_global = np.zeros((n_dof, n_dof))
    P_global = np.zeros(n_dof)

    def get_dof_indices(node_idx):
        return slice(6 * node_idx, 6 * node_idx + 6)
    for elem in elements:
        (i, j) = (elem['node_i'], elem['node_j'])
        (coords_i, coords_j) = (node_coords[i], node_coords[j])
        L = np.linalg.norm(coords_j - coords_i)
        (E, nu) = (elem['E'], elem['nu'])
        (A, Iy, Iz, J, I_rho) = (elem['A'], elem['Iy'], elem['Iz'], elem['J'], elem['I_rho'])
        k_local = local_elastic_stiffness_matrix_3D_beam(E, nu, A, L, Iy, Iz, J)
        T = np.eye(12)
        k_global = T.T @ k_local @ T
        (dofs_i, dofs_j) = (get_dof_indices(i), get_dof_indices(j))
        elem_dofs = np.r_[dofs_i, dofs_j]
        K_global[np.ix_(elem_dofs, elem_dofs)] += k_global
    for (node_idx, loads) in nodal_loads.items():
        dofs = get_dof_indices(node_idx)
        P_global[dofs] += loads
    fixed_dofs = []
    for (node_idx, bc) in boundary_conditions.items():
        node_dofs = list(range(6 * node_idx, 6 * node_idx + 6))
        if isinstance(bc[0], bool):
            for (dof_idx, is_fixed) in enumerate(bc):
                if is_fixed:
                    fixed_dofs.append(node_dofs[dof_idx])
        else:
            for dof_idx in bc:
                fixed_dofs.append(node_dofs[dof_idx])
    free_dofs = [i for i in range(n_dof) if i not in fixed_dofs]
    if not free_dofs:
        raise ValueError('All DOFs are constrained - no free DOFs for analysis')
    K_free = K_global[np.ix_(free_dofs, free_dofs)]
    P_free = P_global[free_dofs]
    try:
        u_free = scipy.linalg.solve(K_free, P_free, assume_a='sym')
    except scipy.linalg.LinAlgError as e:
        raise ValueError(f'Singular or ill-conditioned stiffness matrix: {e}')
    u_global = np.zeros(n_dof)
    u_global[free_dofs] = u_free
    K_g_global = np.zeros((n_dof, n_dof))
    for elem in elements:
        (i, j) = (elem['node_i'], elem['node_j'])
        (coords_i, coords_j) = (node_coords[i], node_coords[j])
        L = np.linalg.norm(coords_j - coords_i)
        (A, I_rho) = (elem['A'], elem['I_rho'])
        (dofs_i, dofs_j) = (get_dof_indices(i), get_dof_indices(j))
        u_elem = np.r_[u_global[dofs_i], u_global[dofs_j]]
        Fx2 = 0.0
        (Mx2, My1, Mz1, My2, Mz2) = (0.0, 0.0, 0.0, 0.0, 0.0)
        k_g_local = local_geometric_stiffness_matrix_3D_beam(L, A, I_rho, Fx2, Mx2, My1, Mz1, My2, Mz2)
        T = np.eye(12)
        k_g_global = T.T @ k_g_local @ T
        elem_dofs = np.r_[dofs_i, dofs_j]
        K_g_global[np.ix_(elem_dofs, elem_dofs)] += k_g_global
    K_g_free = K_g_global[np.ix_(free_dofs, free_dofs)]