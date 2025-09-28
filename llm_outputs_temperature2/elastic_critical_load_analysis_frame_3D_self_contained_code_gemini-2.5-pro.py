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
        if L < 1e-12:
            raise ValueError('Element has zero length.')
        x_local = v / L
        if local_z_vec is not None:
            z_prime = np.array(local_z_vec, dtype=float)
            norm_z_prime = np.linalg.norm(z_prime)
            if norm_z_prime < 1e-09:
                raise ValueError('local_z vector cannot be a zero vector.')
            z_prime /= norm_z_prime
            if abs(np.dot(x_local, z_prime)) > 1.0 - 1e-09:
                raise ValueError('local_z vector cannot be collinear with the element axis.')
            y_local = np.cross(z_prime, x_local)
            y_local /= np.linalg.norm(y_local)
            z_local = np.cross(x_local, y_local)
        else:
            Z_global = np.array([0.0, 0.0, 1.0])
            if abs(np.dot(x_local, Z_global)) > 1.0 - 1e-09:
                Y_global = np.array([0.0, 1.0, 0.0])
                z_local = np.cross(x_local, Y_global)
                z_local /= np.linalg.norm(z_local)
                y_local = np.cross(z_local, x_local)
            else:
                y_local = np.cross(Z_global, x_local)
                y_local /= np.linalg.norm(y_local)
                z_local = np.cross(x_local, y_local)
        R = np.vstack([x_local, y_local, z_local]).T
        T = np.zeros((12, 12))
        for i in range(4):
            T[i * 3:(i + 1) * 3, i * 3:(i + 1) * 3] = R
        return (T, L)

    def _get_local_elastic_stiffness(E, nu, A, Iy, Iz, J, L):
        G = E / (2 * (1 + nu))
        k = np.zeros((12, 12))
        (k[0, 0], k[6, 6]) = (E * A / L, E * A / L)
        (k[0, 6], k[6, 0]) = (-E * A / L, -E * A / L)
        (k[3, 3], k[9, 9]) = (G * J / L, G * J / L)
        (k[3, 9], k[9, 3]) = (-G * J / L, -G * J / L)
        (c1z, c2z, c3z, c4z) = (12 * E * Iz / L ** 3, 6 * E * Iz / L ** 2, 4 * E * Iz / L, 2 * E * Iz / L)
        (k[1, 1], k[7, 7]) = (c1z, c1z)
        (k[1, 7], k[7, 1]) = (-c1z, -c1z)
        (k[1, 5], k[5, 1]) = (c2z, c2z)
        (k[1, 11], k[11, 1]) = (c2z, c2z)
        (k[7, 5], k[5, 7]) = (-c2z, -c2z)
        (k[7, 11], k[11, 7]) = (-c2z, -c2z)
        (k[5, 5], k[11, 11]) = (c3z, c3z)
        (k[5, 11], k[11, 5]) = (c4z, c4z)
        (c1y, c2y, c3y, c4y) = (12 * E * Iy / L ** 3, 6 * E * Iy / L ** 2, 4 * E * Iy / L, 2 * E * Iy / L)
        (k[2, 2], k[8, 8]) = (c1y, c1y)
        (k[2, 8], k[8, 2]) = (-c1y, -c1y)
        (k[2, 4], k[4, 2]) = (-c2y, -c2y)
        (k[2, 10], k[10, 2]) = (-c2y, -c2y)
        (k[8, 4], k[4, 8]) = (c2y, c2y)
        (k[8, 10], k[10, 8]) = (c2y, c2y)
        (k[4, 4], k[10, 10]) = (c3y, c3y)
        (k[4, 10], k[10, 4]) = (c4y, c4y)
        return k

    def _get_local_geometric_stiffness(N, I_rho, A, L):
        kg = np.zeros((12, 12))
        c1 = 6 / 5 * N / L
        c2 = N / 10
        c3 = 2 * L / 15 * N
        c4 = -L / 30 * N
        (kg[1, 1], kg[7, 7]) = (c1, c1)
        (kg[1, 7], kg[7, 1]) = (-c1, -c1)
        (kg[2, 2], kg[8, 8]) = (c1, c1)
        (kg[2, 8], kg[8, 2]) = (-c1, -c1)
        (kg[1, 5], kg[5, 1]) = (c2, c2)
        (kg[1, 11], kg[11, 1]) = (c2, c2)
        (kg[7, 5], kg[5, 7]) = (-c2, -c2)
        (kg[7, 11], kg[11, 7]) = (-c2, -c2)
        (kg[2, 4], kg[4, 2]) = (-c2, -c2)
        (kg[2, 10], kg[10, 2]) = (-c2, -c2)
        (kg[8, 4], kg[4, 8]) = (c2, c2)
        (kg[8, 10], kg[10, 8]) = (c2, c2)
        (kg[4, 4], kg[10, 10]) = (c3, c3)
        (kg[4, 10], kg[10, 4]) = (c4, c4)
        (kg[5, 5], kg[11, 11]) = (c3, c3)
        (kg[5, 11], kg[11, 5]) = (c4, c4)
        if A > 1e-12:
            c_torsion = N * I_rho / A
            (kg[3, 3], kg[9, 9]) = (c_torsion, c_torsion)
            (kg[3, 9], kg[9, 3]) = (-c_torsion, -c_torsion)
        return kg
    n_nodes = node_coords.shape[0]
    n_dof = n_nodes * 6
    K = np.zeros((n_dof, n_dof))
    P = np.zeros(n_dof)
    element_transforms = []
    for el in elements:
        (node_i, node_j) = (el['node_i'], el['node_j'])
        (p1, p2) = node