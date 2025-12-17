def MSA_3D_elastic_critical_load_CC1_H10_T1(node_coords: np.ndarray, elements: Sequence[dict], boundary_conditions: dict[int, Sequence[int]], nodal_loads: dict[int, Sequence[float]]):
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
    node_coords : (N, 3) float ndarray
        Global coordinates of the N nodes (row i → [x, y, z] of node i, 0-based).
    elements : iterable of dict
        Each dict must contain:
            'node_i', 'node_j' : int
                End node indices (0-based).
            'E', 'nu', 'A', 'I_y', 'I_z', 'J' : float
                Material and geometric properties.
            'local_z' : array-like of shape (3,), optional
                Unit vector in global coordinates defining the local z-axis orientation.
                Must not be parallel to the beam axis. If None, a default is chosen.
                Default local_z = global z, unless the beam lies along global z — then default local_z = global y.
    boundary_conditions : dict[int, Sequence[int]]
        Maps node index to a 6-element iterable of 0 (free) or 1 (fixed) values.
        Omitted nodes are assumed to have all DOFs free.
    nodal_loads : dict[int, Sequence[float]]
        Maps node index to a 6-element array of applied loads:
        [Fx, Fy, Fz, Mx, My, Mz]. Omitted nodes are assumed to have zero loads.
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
    Helper Functions (used here)
    ----------------------------
        Builds the global elastic stiffness `K` with shape `(6*n_nodes, 6*n_nodes)`.
        Builds the global reference load vector `P` with shape `(6*n_nodes,)`.
        Returns `(fixed, free)` DOF indices given `boundary_conditions`.
        Solves the constrained linear system for displacements; returns
        `(u_global, reactions_or_aux)`.
        Builds the global geometric stiffness `K_g` (same shape as `K`), using
        the displacement state `u_global` from the linear solve.
        Solves `K φ = -λ K_g φ` on free DOFs, selects the smallest positive `λ`,
        and embeds the mode back into a full global vector.
    Raises
    ------
    ValueError
        Propagated from called routines if:
    Notes
    -----
      if desired (e.g., by max absolute translational DOF).
      the returned mode can depend on numerical details.
        + **Tension (+Fx2)** increases lateral/torsional stiffness.
        + Compression (-Fx2) decreases it and may trigger buckling when K_e + K_g becomes singular.
    """
    node_coords = np.asarray(node_coords, dtype=float)
    n_nodes = int(node_coords.shape[0])
    bc_bool = {}
    for n, flags in (boundary_conditions or {}).items():
        arr = np.asarray(flags, dtype=bool).ravel()
        if arr.size != 6:
            raise ValueError(f'Boundary conditions for node {n} must have 6 entries.')
        bc_bool[int(n)] = arr
    K = assemble_global_stiffness_matrix_linear_elastic_3D(node_coords, elements)
    P = assemble_global_load_vector_linear_elastic_3D(nodal_loads or {}, n_nodes)
    fixed, free = partition_degrees_of_freedom(bc_bool, n_nodes)
    u_global, _ = linear_solve(P, K, fixed, free)
    elements_g = []
    for ele in elements:
        e = dict(ele)
        if 'I_rho' not in e:
            if 'J' in e and e['J'] is not None:
                try:
                    e['I_rho'] = float(e['J'])
                except Exception:
                    e['I_rho'] = float(e.get('I_y', 0.0)) + float(e.get('I_z', 0.0))
            else:
                e['I_rho'] = float(e.get('I_y', 0.0)) + float(e.get('I_z', 0.0))
        elements_g.append(e)
    K_g = assemble_global_geometric_stiffness_3D_beam(node_coords, elements_g, u_global)
    return eigenvalue_analysis(K, K_g, bc_bool, n_nodes)