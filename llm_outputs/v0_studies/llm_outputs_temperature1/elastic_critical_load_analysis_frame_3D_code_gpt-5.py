def elastic_critical_load_analysis_frame_3D(node_coords: np.ndarray, elements: Sequence[dict], boundary_conditions: dict[int, Sequence[int]], nodal_loads: dict[int, Sequence[float]]):
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
    (Downstream) Helpers:
    -----------------------------------------------------------------------
    These are not called directly here but are dependencies of:
    `assemble_global_geometric_stiffness_3D_beam`:
    `assemble_global_stiffness_matrix_linear_elastic_3D`:
    Raises
    ------
    ValueError
        Propagated from called routines if:
    Notes
    -----
      if desired (e.g., by max absolute translational DOF).
      the returned mode can depend on numerical details.
    """
    node_coords = np.asarray(node_coords, dtype=float)
    if node_coords.ndim != 2 or node_coords.shape[1] != 3:
        raise ValueError('node_coords must be an array of shape (n_nodes, 3).')
    n_nodes = node_coords.shape[0]
    n_dof = 6 * n_nodes
    bc_bool: dict[int, np.ndarray] = {}
    if boundary_conditions is not None:
        for (n, spec) in boundary_conditions.items():
            ni = int(n)
            if ni < 0 or ni >= n_nodes:
                raise ValueError(f'Boundary condition specified for invalid node index {ni}.')
            try:
                seq = list(spec)
            except TypeError:
                raise ValueError(f'Boundary condition for node {ni} must be sequence-like.')
            if len(seq) == 6:
                mask = np.asarray(seq, dtype=bool)
            else:
                mask = np.zeros(6, dtype=bool)
                for idx in seq:
                    ii = int(idx)
                    if ii < 0 or ii >= 6:
                        raise ValueError(f'Invalid DOF index {ii} in boundary condition for node {ni}.')
                    mask[ii] = True
            bc_bool[ni] = mask
    if nodal_loads is None:
        nodal_loads = {}
    else:
        for n in nodal_loads.keys():
            ni = int(n)
            if ni < 0 or ni >= n_nodes:
                raise ValueError(f'Nodal load specified for invalid node index {ni}.')
    K_e = assemble_global_stiffness_matrix_linear_elastic_3D(node_coords, elements)
    if K_e.shape != (n_dof, n_dof):
        raise ValueError('Assembled elastic stiffness has incorrect shape.')
    P = assemble_global_load_vector_linear_elastic_3D(nodal_loads, n_nodes)
    if P.shape != (n_dof,):
        raise ValueError('Assembled global load vector has incorrect shape.')
    (fixed, free) = partition_degrees_of_freedom(bc_bool, n_nodes)
    if free.size == 0:
        raise ValueError('No free degrees of freedom remain after applying boundary conditions.')
    (u_ref, _) = linear_solve(P, K_e, fixed, free)
    K_g = assemble_global_geometric_stiffness_3D_beam(node_coords, elements, u_ref)
    if K_g.shape != (n_dof, n_dof):
        raise ValueError('Assembled geometric stiffness has incorrect shape.')
    (lam, phi) = eigenvalue_analysis(K_e, K_g, bc_bool, n_nodes)
    return (lam, phi)