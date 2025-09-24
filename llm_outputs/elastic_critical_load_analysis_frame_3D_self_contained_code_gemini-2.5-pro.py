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
    Raises
    ------
    ValueError
        Propagated from called routines if:
    Notes
    -----
      if desired (e.g., by max absolute translational DOF).
      the returned mode can depend on numerical details.
    """

    def _get_transformation_matrix(p1, p2, local_z_vec):
        v = p2 - p1
        L = np.linalg.norm(v)
        if L < 1e-09:
            raise ValueError('Element length is close to zero.')
        x_vec = v / L
        if local_z_vec is not None:
            z_user = np.array(local_z_vec, dtype=float)
            y_vec = np.cross(z_user, x_vec)
            norm_y = np.linalg.norm(y_vec)
            if norm_y < 1e-09:
                raise ValueError('local_z cannot be parallel to the element axis.')
            y_vec /= norm_y
            z_vec = np.cross(x_vec, y_vec)
        else:
            Z_glob = np.array([0.0, 0.0, 1.0])
            if np.isclose(np.abs(np.dot(x_vec, Z_glob)), 1.0):
                Y_glob = np.array([0.0, 1.0, 0.0])
                y_vec = np.cross(Y_glob, x_vec)
                y_vec /= np.linalg.norm(y_vec)
                z_vec = np.cross(x_vec, y_vec)
            else:
                y_vec = np.cross(Z_glob, x_vec)
                y_vec /= np.linalg.norm(y_vec)
                z_vec = np.cross(x_vec, y_vec)
        R = np.vstack([x_vec, y_vec, z_vec])
        T_lambda = scipy.linalg.block_diag(R, R, R, R)
        return (T_lambda, L)

    def _get_elastic_stiffness_matrix_local(E, G, A, Iy, Iz, J, L):
        (L2, L3) = (L * L, L * L * L)
        k = np.zeros((12, 12))
        EA_L = E * A / L
        GJ_L = G * J / L
        EIz_L = E * Iz / L
        EIz_L2 = E * Iz / L2
        EIz_L3 = E * Iz / L3
        EIy_L = E * Iy / L
        EIy_L2 = E * Iy / L2
        EIy_L3 = E * Iy / L3
        (k[0, 0], k[6, 6]) = (EA_L, EA_L)
        (k[0, 6], k[6, 0]) = (-EA_L, -EA_L)
        (k[3, 3], k[9, 9]) = (GJ_L, GJ_L)
        (k[3, 9], k[9, 3]) = (-GJ_L, -GJ_L)
        (k[1, 1], k[7, 7]) = (12 * EIz_L3, 12 * EIz_L3)
        (k[1, 7], k[7, 1]) = (-12 * EIz_L3, -12 * EIz_L3)
        (k[1, 5], k[5, 1]) = (6 * EIz_L2, 6 * EIz_L2)
        (k[1, 11], k[11, 1]) = (6 * EIz_L2, 6 * EIz_L2)
        (k[7, 5], k[5, 7]) = (-6 * EIz_L2, -6 * EIz_L2)
        (k[7, 11], k[11, 7]) = (-6 * EIz_L2, -6 * EIz_L2)
        (k[5, 5], k[11, 11]) = (4 * EIz_L, 4 * EIz_L)
        (k[5, 11], k[11, 5]) = (2 * EIz_L, 2 * EIz_L)
        (k[2, 2], k[8, 8]) = (12 * EIy_L3, 12 * EIy_L3)
        (k[2, 8], k[8, 2]) = (-12 * EIy_L3, -12 * EIy_L3)
        (k[2, 4], k[4, 2]) = (-6 * EIy_L2, -6 * EIy_L2)
        (k[2, 10], k[10, 2]) = (-6 * EIy_L2, -6 * EIy_L2)
        (k[8, 4], k[4, 8]) = (6 * EIy_L2, 6 * EIy_L2)
        (k[8, 10], k[10, 8]) = (6 * EIy_L2, 6 * EIy_L2)
        (k[4, 4], k[10, 10]) = (4 * EIy_L, 4 * EIy_L)
        (k[4, 10], k[10, 4]) = (2 * EIy_L, 2 * EIy_L)
        return k

    def _get_geometric_stiffness_matrix_local(N, L, I_rho_over_A):
        kg = np.zeros((12, 12))
        P_30L = N / (30 * L)
        kg_bend = P_30L * np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 36, 0, 0, 0, 3 * L, 0, -36, 0, 0, 0, 3 * L], [0, 0, 36, 0, -3 * L, 0, 0, 0, -36, 0, -3 * L, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, -3 * L, 0, 4 * L * L, 0, 0, 0, 3 * L, 0, -L * L, 0], [0, 3 * L, 0, 0, 0, 4 * L * L, 0, -3 * L, 0, 0, 0, -L * L], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, -36, 0, 0, 0, -3 * L, 0, 36, 0, 0, 0, -3 * L], [0, 0, -36, 0, 3 * L, 0, 0, 0, 36, 0, 3 * L, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, -3 * L, 0, -L * L, 0, 0, 0, 3 * L, 0, 4 * L * L, 0], [0, 3 * L, 0, 0, 0, -L * L, 0, -3 * L, 0, 0, 0, 4 * L * L]])
        P_Irho_A = N * I_rho_