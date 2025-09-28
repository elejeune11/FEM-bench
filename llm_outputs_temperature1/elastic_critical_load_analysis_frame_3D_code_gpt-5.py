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
    n_nodes = int(node_coords.shape[0])

    def _normalize_bcs(bc_dict, n_nodes):
        bc_bool = {}
        for (n, spec) in (bc_dict or {}).items():
            arr = np.asarray(spec)
            if arr.dtype == bool and arr.size == 6:
                bc_bool[int(n)] = [bool(x) for x in arr.tolist()]
            elif arr.size == 6 and np.issubdtype(arr.dtype, np.integer) and np.all((arr == 0) | (arr == 1)):
                bc_bool[int(n)] = [bool(x) for x in arr.tolist()]
            else:
                idx = [int(i) for i in np.asarray(spec, dtype=int).tolist()]
                flags = [False] * 6
                for i in idx:
                    if i < 0 or i >= 6:
                        raise ValueError(f'Boundary condition DOF index out of range at node {n}: {i}')
                    flags[i] = True
                bc_bool[int(n)] = flags
        return bc_bool

    def _normalize_elements(elems):
        out = []
        for e in elems:
            d = dict(e)
            if 'I_y' not in d and 'Iy' in d:
                d['I_y'] = d['Iy']
            if 'I_z' not in d and 'Iz' in d:
                d['I_z'] = d['Iz']
            out.append(d)
        return out
    elements_norm = _normalize_elements(elements)
    bc_bool = _normalize_bcs(boundary_conditions, n_nodes)
    K = assemble_global_stiffness_matrix_linear_elastic_3D(node_coords, elements_norm)
    P = assemble_global_load_vector_linear_elastic_3D(nodal_loads, n_nodes)
    (fixed, free) = partition_degrees_of_freedom(bc_bool, n_nodes)
    (u, _) = linear_solve(P, K, fixed, free)
    K_g = assemble_global_geometric_stiffness_3D_beam(node_coords, elements_norm, u)
    (lam, mode) = eigenvalue_analysis(K, K_g, bc_bool, n_nodes)
    return (lam, mode)