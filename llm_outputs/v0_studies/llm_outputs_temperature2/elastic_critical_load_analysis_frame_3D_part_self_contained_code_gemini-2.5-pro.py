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
    n_nodes = len(node_coords)
    n_dof = n_nodes * 6
    K = np.zeros((n_dof, n_dof))
    element_data_cache = []
    for el in elements:
        (node_i, node_j) = (el['node_i'], el['node_j'])
        (coord_i, coord_j) = (node_coords[node_i], node_coords[node_j])
        vec_ij = coord_j - coord_i
        L = np.linalg.norm(vec_ij)
        if L < 1e-09:
            raise ValueError(f'Element between nodes {node_i} and {node_j} has zero length.')
        x_local = vec_ij / L
        if el['local_z'] is not None:
            z_trial = np.array(el['local_z'], dtype=float)
            norm_z_trial = np.linalg.norm(z_trial)
            if norm_z_trial < 1e-09:
                raise ValueError('local_z vector cannot be a zero vector.')
            z_trial /= norm_z_trial
            if np.allclose(np.abs(np.dot(x_local, z_trial)), 1.0):
                raise ValueError('local_z vector cannot be collinear with the element axis.')
            y_local = np.cross(z_trial, x_local)
            y_local /= np.linalg.norm(y_local)
            z_local = np.cross(x_local, y_local)
        else:
            V = np.array([0.0, 0.0, 1.0])
            if np.allclose(np.abs(np.dot(x_local, V)), 1.0):
                V = np.array([0.0, 1.0, 0.0])
            y_local = np.cross(V, x_local)
            y_local /= np.linalg.norm(y_local)
            z_local = np.cross(x_local, y_local)
        R = np.vstack([x_local, y_local, z_local])
        T = np.kron(np.eye(4), R)
        k_e = local_elastic_stiffness_matrix_3D_beam(el['E'], el['nu'], el['A'], L, el['Iy'], el['Iz'], el['J'])
        K_e_global = T.T @ k_e @ T
        dof_indices = np.concatenate([np.arange(node_i * 6, node_i * 6 + 6), np.arange(node_j * 6, node_j * 6 + 6)])
        K[np.ix_(dof_indices, dof_indices)] += K_e_global
        element_data_cache.append({'L': L, 'T': T, 'k_e': k_e, 'dof_indices': dof_indices})
    P = np.zeros(n_dof)
    for (node_idx, load_vec) in nodal_loads.items():
        P[node_idx * 6:node_idx * 6 + 6] += load_vec
    all_dofs = np.arange(n_dof)
    fixed_dofs = set()
    for (node_idx, bc_spec) in boundary_conditions.items():
        start_dof = node_idx * 6
        if isinstance(bc_spec[0], (bool, np.bool_)):
            fixed_dofs.update({start_dof + i for (i, is_fixed) in enumerate(bc_spec) if is_fixed})
        else:
            fixed_dofs.update({start_dof + i for i in bc_spec})
    fixed_dofs = sorted(list(fixed_dofs))
    free_dofs = np.setdiff1d(all_dofs, fixed_dofs, assume_unique=True)
    if len(free_dofs) == 0:
        raise ValueError('No free degrees of freedom.')
    K_free = K[np.ix_(free_dofs, free_dofs)]
    P_free = P[free_dofs]
    try:
        u_free = np.linalg.solve(K_free, P_free)
    except np.linalg.LinAlgError:
        raise ValueError('The reduced stiffness matrix is singular. Check boundary conditions for rigid-body modes.')
    u = np.zeros(n_dof)
    u[free_dofs] = u_free
    K_g = np.zeros((n_dof, n_dof))
    for (i, el) in enumerate(elements):
        cache = element_data_cache[i]
        u_e_global = u[cache['dof_indices']]
        u_e_local = cache['T'] @ u_e_global
        f_e_local = cache['k_e'] @ u_e_local
        k_g_local = local_geometric_stiffness_matrix_3D_beam(L=cache['L'], A=el['A'], I_rho=el['I_rho'], Fx2=f_e_local[6], Mx2=f_e_local[9], My1=f_e_local[4], Mz1=f_e_local[5], My2=f_e_local[10], Mz2=f_e_local[11])
        K_g_global = cache['T'].T @ k_g_local @ cache['T']
        K_g[np.ix_(cache['dof_indices'], cache['dof_indices'])] += K_g_global
    Kg_free = K_g[np.ix_(free_dofs, free_dofs)]
    try:
        (eigenvalues, eigenvectors) = scipy.linalg.eigh(K_free, -Kg_free)
    except (np.linalg.LinAlgError, scipy.linalg.LinAlgError) as e:
        raise ValueError(f'Eigenvalue problem failed to converge: {e}')
    if np.iscomplexobj(eigenvalues):
        if not np.allclose(np.imag(eigenvalues), 0):
            raise ValueError('Eigenvalues have significant imaginary parts.')
        eigenvalues = np.real(eigenvalues)
    positive_mask = eigenvalues > 1e-09
    if not np.any(positive_mask):
        raise ValueError('No positive eigenvalue found, indicating no buckling load or an unstable initial state.')
    positive_eigenvalues = eigenvalues[positive_mask]
    min_lambda_idx_in_positive = np.argmin(positive_eigenvalues)
    min_lambda = positive_eigenvalues[min_lambda_idx_in_positive]
    original_indices = np.where(positive_mask)[0]
    min_lambda_original_idx = original_indices[min_lambda_idx_in_positive]
    phi_free = eigenvectors[:, min_lambda_original_idx]
    deformed_shape_vector = np.zeros(n_dof)
    deformed_shape_vector[free_dofs] = phi_free
    elastic_critical_load_factor = min_lambda
    return (elastic_critical_load_factor, deformed_shape_vector)