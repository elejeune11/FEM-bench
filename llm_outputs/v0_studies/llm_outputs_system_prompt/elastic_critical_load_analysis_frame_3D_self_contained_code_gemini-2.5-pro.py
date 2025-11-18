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

    def _get_transformation_matrix_3d(node_i_coords, node_j_coords, local_z_vec):
        vec_ij = node_j_coords - node_i_coords
        L = np.linalg.norm(vec_ij)
        if L < 1e-09:
            raise ValueError('Element length is close to zero.')
        x_local = vec_ij / L
        if local_z_vec is not None:
            ref_vec = np.array(local_z_vec, dtype=float)
            z_local = ref_vec - np.dot(ref_vec, x_local) * x_local
            if np.linalg.norm(z_local) < 1e-09:
                raise ValueError('local_z vector cannot be parallel to the element axis.')
            z_local /= np.linalg.norm(z_local)
            y_local = np.cross(z_local, x_local)
        elif np.allclose(np.abs(x_local), [0, 0, 1]):
            ref_vec = np.array([1.0, 0.0, 0.0])
            y_local = np.cross(x_local, ref_vec)
            y_local /= np.linalg.norm(y_local)
            z_local = np.cross(x_local, y_local)
        else:
            global_z = np.array([0.0, 0.0, 1.0])
            y_local = np.cross(global_z, x_local)
            y_local /= np.linalg.norm(y_local)
            z_local = np.cross(x_local, y_local)
        R = np.array([x_local, y_local, z_local])
        T = np.zeros((12, 12))
        for i in range(4):
            T[i * 3:(i + 1) * 3, i * 3:(i + 1) * 3] = R
        return (T, L)

    def _get_local_elastic_stiffness_matrix_3d(L, E, nu, A, Iy, Iz, J):
        G = E / (2 * (1 + nu))
        k = np.zeros((12, 12))
        k[0, 0] = k[6, 6] = E * A / L
        k[0, 6] = k[6, 0] = -E * A / L
        k[3, 3] = k[9, 9] = G * J / L
        k[3, 9] = k[9, 3] = -G * J / L
        k_b_y = E * Iy / L ** 3 * np.array([[12, -6 * L, -12, -6 * L], [-6 * L, 4 * L ** 2, 6 * L, 2 * L ** 2], [-12, 6 * L, 12, 6 * L], [-6 * L, 2 * L ** 2, 6 * L, 4 * L ** 2]])
        dofs_y = np.ix_([2, 4, 8, 10], [2, 4, 8, 10])
        k[dofs_y] += k_b_y
        k_b_z = E * Iz / L ** 3 * np.array([[12, 6 * L, -12, 6 * L], [6 * L, 4 * L ** 2, -6 * L, 2 * L ** 2], [-12, -6 * L, 12, -6 * L], [6 * L, 2 * L ** 2, -6 * L, 4 * L ** 2]])
        dofs_z = np.ix_([1, 5, 7, 11], [1, 5, 7, 11])
        k[dofs_z] += k_b_z
        return k

    def _get_local_geometric_stiffness_matrix_3d(N, L, I_rho, A):
        k_g = np.zeros((12, 12))
        c1 = N / (30 * L)
        k_g_bend_z = c1 * np.array([[36, 3 * L, -36, 3 * L], [3 * L, 4 * L ** 2, -3 * L, -L ** 2], [-36, -3 * L, 36, -3 * L], [3 * L, -L ** 2, -3 * L, 4 * L ** 2]])
        k_g_bend_y = c1 * np.array([[36, -3 * L, -36, -3 * L], [-3 * L, 4 * L ** 2, 3 * L, -L ** 2], [-36, 3 * L, 36, 3 * L], [-3 * L, -L ** 2, 3 * L, 4 * L ** 2]])
        dofs_z = np.ix_([2, 4, 8, 10], [2, 4, 8, 10])
        dofs_y = np.ix_([1, 5, 7, 11], [1, 5, 7, 11])
        k_g[dofs_z] = k_g_bend_z
        k_g[dofs_y] = k_g_bend_y
        if A > 1e-12:
            c2 = N * I_rho / (A * L)
            k_g_torsion = c2 * np.array([[1, -1], [-1, 1]])
            dofs_x_rot = np.ix_([3, 9], [3, 9])
            k_g[dofs_x_rot] = k_g_torsion
        return k_g
    n_nodes = len(node_coords)
    n_dof = n_nodes * 6
    K = np.zeros((n_dof, n_dof))
    P = np.zeros(n_dof)
    element_cache = []
    for elem in elements:
        (i, j) = (elem['node_i'], elem['node_j'])
        node_i_coords = node_coords[i]
        node_j_coords = node_coords[j]
        (T, L) = _get_transformation_matrix_3d(node_i_coords, node_j_coords, elem.get('local_z'))
        k_e = _get_local_elastic_stiffness_matrix_3d(L, elem['E'], elem['nu'], elem['A'], elem['Iy'], elem['Iz'], elem['J'])
        K_e = T.T @ k_e @ T
        dof_indices = np.concatenate((np.arange(6 * i, 6 * i + 6), np.arange(6 * j, 6 * j + 6)))
        K[np.ix_(dof_indices, dof_indices)] += K_e
        element_cache.append({'T': T, 'L': L})
    for (node_idx, loads) in nodal_loads.items():
        P[6 * node_idx:6 * node_idx + 6] += loads
    constrained_dofs = np.zeros(n_dof, dtype=bool)
    for (node_idx, bc_spec) in boundary_conditions.items():
        start_dof = 6 * node_idx
        if all((isinstance(x, bool) for x in bc_spec)):
            for i in range(6):
                if bc_spec[i]:
                    constrained_dofs[start_dof + i] = True
        else:
            for dof_idx in bc_spec:
                constrained_dofs[start_dof + dof_idx] = True
    free_dofs = np.where(~constrained_dofs)[0]
    K_free = K[np.ix_(free_dofs, free_dofs)]
    P_free = P[free_dofs]
    try:
        u_free = scipy.linalg.solve(K_free, P_free, assume_a='sym')
    except scipy.linalg.LinAlgError:
        raise ValueError('The stiffness matrix is singular. Check boundary conditions for rigid body modes.')
    u = np.zeros(n_dof)
    u[free_dofs] = u_free
    K_g = np.zeros((n_dof, n_dof))
    for (idx, elem) in enumerate(elements):
        (i, j) = (elem['node_i'], elem['node_j'])
        T = element_cache[idx]['T']
        L = element_cache[idx]['L']
        dof_indices = np.concatenate((np.arange(6 * i, 6 * i + 6), np.arange(6 * j, 6 * j + 6)))
        u_e_global = u[dof_indices]
        u_e_local = T @ u_e_global
        N = elem['E'] * elem['A'] / L * (u_e_local[6] - u_e_local[0])
        k_g = _get_local_geometric_stiffness_matrix_3d(N, L, elem['I_rho'], elem['A'])
        K_g_e = T.T @ k_g @ T
        K_g[np.ix_(dof_indices, dof_indices)] += K_g_e
    K_g_free = K_g[np.ix_(free_dofs, free_dofs)]
    try:
        (eigenvalues, eigenvectors) = scipy.linalg.eigh(K_free, b=-K_g_free)
    except scipy.linalg.LinAlgError:
        raise ValueError('Eigenvalue problem failed to solve. The geometric stiffness matrix may be singular or ill-conditioned.')
    if np.iscomplexobj(eigenvalues):
        raise ValueError('Complex eigenvalues found, indicating a non-physical or unstable system.')
    positive_mask = eigenvalues > 1e-09
    if not np.any(positive_mask):
        raise ValueError('No positive eigenvalue found. The structure may be unstable under the reference load or the load does not induce buckling.')
    positive_eigenvalues = eigenvalues[positive_mask]
    elastic_critical_load_factor = positive_eigenvalues[0]
    original_idx = np.where(eigenvalues == elastic_critical_load_factor)[0][0]
    phi_free = eigenvectors[:, original_idx]
    deformed_shape_vector = np.zeros(n_dof)
    deformed_shape_vector[free_dofs] = phi_free
    return (float(elastic_critical_load_factor), deformed_shape_vector)