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
    n_dof = n_nodes * 6
    K = np.zeros((n_dof, n_dof))
    K_g = np.zeros((n_dof, n_dof))
    P = np.zeros(n_dof)
    element_data_for_kg = []
    for el in elements:
        (i, j) = (el['node_i'], el['node_j'])
        (coord_i, coord_j) = (node_coords[i], node_coords[j])
        vec_ij = coord_j - coord_i
        L = np.linalg.norm(vec_ij)
        if L < 1e-12:
            continue
        local_x = vec_ij / L
        local_z_dir = el.get('local_z')
        if local_z_dir is None:
            if abs(local_x[2]) > 0.9999:
                ref_vec = np.array([0.0, 1.0, 0.0])
            else:
                ref_vec = np.array([0.0, 0.0, 1.0])
            if np.linalg.norm(np.cross(local_x, ref_vec)) < 1e-06:
                ref_vec = np.array([0.0, 1.0, 0.0]) if abs(local_x[1]) < 0.9999 else np.array([1.0, 0.0, 0.0])
            temp_y = np.cross(ref_vec, local_x)
            local_y = temp_y / np.linalg.norm(temp_y)
            local_z = np.cross(local_x, local_y)
        else:
            local_z_dir_vec = np.array(local_z_dir)
            temp_y = np.cross(local_z_dir_vec, local_x)
            if np.linalg.norm(temp_y) < 1e-06:
                raise ValueError(f'local_z for element {i}-{j} is parallel to the element axis.')
            local_y = temp_y / np.linalg.norm(temp_y)
            local_z = np.cross(local_x, local_y)
        R = np.vstack((local_x, local_y, local_z)).T
        T = np.kron(np.eye(4, dtype=float), R)
        k_e = local_elastic_stiffness_matrix_3D_beam(el['E'], el['nu'], el['A'], L, el['Iy'], el['Iz'], el['J'])
        K_e_global = T @ k_e @ T.T
        dof_indices = np.r_[6 * i:6 * i + 6, 6 * j:6 * j + 6]
        K[np.ix_(dof_indices, dof_indices)] += K_e_global
        element_data_for_kg.append({'k_e': k_e, 'T': T, 'L': L, 'A': el['A'], 'I_rho': el['I_rho'], 'dof_indices': dof_indices})
    for (node_idx, load_vec) in nodal_loads.items():
        start_dof = 6 * node_idx
        P[start_dof:start_dof + 6] += load_vec
    constrained_dofs_mask = np.zeros(n_dof, dtype=bool)
    for (node_idx, bc_spec) in boundary_conditions.items():
        start_dof = 6 * node_idx
        if bc_spec and isinstance(bc_spec[0], bool):
            for i in range(6):
                if bc_spec[i]:
                    constrained_dofs_mask[start_dof + i] = True
        else:
            for dof_idx in bc_spec:
                constrained_dofs_mask[start_dof + int(dof_idx)] = True
    free_dofs = np.where(~constrained_dofs_mask)[0]
    if free_dofs.size == 0:
        raise ValueError('No free degrees of freedom.')
    K_ff = K[np.ix_(free_dofs, free_dofs)]
    P_f = P[free_dofs]
    try:
        u_f = scipy.linalg.solve(K_ff, P_f, assume_a='sym')
    except np.linalg.LinAlgError:
        raise ValueError('The stiffness matrix is singular. Check boundary conditions for rigid-body modes.')
    u = np.zeros(n_dof)
    u[free_dofs] = u_f
    for (i, el_data) in enumerate(element_data_for_kg):
        T = el_data['T']
        k_e = el_data['k_e']
        dof_indices = el_data['dof_indices']
        u_e_global = u[dof_indices]
        u_e_local = T.T @ u_e_global
        f_e_local = k_e @ u_e_local
        Fx2 = f_e_local[6]
        Mx2 = f_e_local[9]
        My1 = f_e_local[4]
        Mz1 = f_e_local[5]
        My2 = f_e_local[10]
        Mz2 = f_e_local[11]
        k_g_local = local_geometric_stiffness_matrix_3D_beam(el_data['L'], el_data['A'], el_data['I_rho'], Fx2, Mx2, My1, Mz1, My2, Mz2)
        K_g_global = T @ k_g_local @ T.T
        K_g[np.ix_(dof_indices, dof_indices)] += K_g_global
    K_g_ff = K_g[np.ix_(free_dofs, free_dofs)]
    try:
        (eigenvalues, eigenvectors) = scipy.linalg.eig(K_ff, -K_g_ff)
    except (np.linalg.LinAlgError, scipy.linalg.LinAlgError):
        raise ValueError('Eigenvalue problem failed to solve. Check matrices for singularity or conditioning.')
    positive_real_eigenvalues = []
    eigenvalue_indices = []
    TOL = 1e-09
    for (i, val) in enumerate(eigenvalues):
        if abs(np.imag(val)) > TOL * abs(np.real(val)):
            continue
        real_val = np.real(val)
        if real_val > TOL:
            positive_real_eigenvalues.append(real_val)
            eigenvalue_indices.append(i)
    if not positive_real_eigenvalues:
        raise ValueError('No positive eigenvalue found for buckling analysis.')
    min_idx_in_pos_list = np.argmin(positive_real_eigenvalues)
    elastic_critical_load_factor = positive_real_eigenvalues[min_idx_in_pos_list]
    original_idx = eigenvalue_indices[min_idx_in_pos_list]
    buckling_mode_f = eigenvectors[:, original_idx]
    deformed_shape_vector = np.zeros(n_dof, dtype=float)
    deformed_shape_vector[free_dofs] = np.real(buckling_mode_f)
    return (elastic_critical_load_factor, deformed_shape_vector)