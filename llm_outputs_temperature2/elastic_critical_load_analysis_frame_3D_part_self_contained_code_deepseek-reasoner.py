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
        Mapping from node index -> length-6 vector of load components applied at
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
    n_dofs = 6 * n_nodes
    K_global = np.zeros((n_dofs, n_dofs))
    for elem in elements:
        i = elem['node_i']
        j = elem['node_j']
        coords_i = node_coords[i]
        coords_j = node_coords[j]
        L = np.linalg.norm(coords_j - coords_i)
        vx = (coords_j - coords_i) / L
        if elem.get('local_z') is not None:
            vz = np.array(elem['local_z'])
            vz = vz / np.linalg.norm(vz)
            vy = np.cross(vz, vx)
            vy = vy / np.linalg.norm(vy)
            vz = np.cross(vx, vy)
        else:
            if abs(vx[2]) > 0.9:
                vy = np.array([0.0, 1.0, 0.0])
            else:
                vy = np.array([0.0, 0.0, 1.0])
            vy = vy - np.dot(vy, vx) * vx
            vy = vy / np.linalg.norm(vy)
            vz = np.cross(vx, vy)
        R = np.column_stack([vx, vy, vz])
        T = np.zeros((12, 12))
        for k in range(4):
            T[3 * k:3 * k + 3, 3 * k:3 * k + 3] = R
        k_local = local_elastic_stiffness_matrix_3D_beam(elem['E'], elem['nu'], elem['A'], L, elem['Iy'], elem['Iz'], elem['J'])
        k_global = T.T @ k_local @ T
        dofs = [6 * i, 6 * i + 1, 6 * i + 2, 6 * i + 3, 6 * i + 4, 6 * i + 5, 6 * j, 6 * j + 1, 6 * j + 2, 6 * j + 3, 6 * j + 4, 6 * j + 5]
        for (idx_i, dof_i) in enumerate(dofs):
            for (idx_j, dof_j) in enumerate(dofs):
                K_global[dof_i, dof_j] += k_global[idx_i, idx_j]
    P_global = np.zeros(n_dofs)
    for (node_idx, load) in nodal_loads.items():
        start_dof = 6 * node_idx
        P_global[start_dof:start_dof + 6] = load
    fixed_dofs = set()
    for (node_idx, bc) in boundary_conditions.items():
        start_dof = 6 * node_idx
        if isinstance(bc[0], bool):
            for (dof_idx, is_fixed) in enumerate(bc):
                if is_fixed:
                    fixed_dofs.add(start_dof + dof_idx)
        else:
            for dof_idx in bc:
                fixed_dofs.add(start_dof + dof_idx)
    free_dofs = [i for i in range(n_dofs) if i not in fixed_dofs]
    if not free_dofs:
        raise ValueError('No free DOFs remaining after applying boundary conditions')
    K_ff = K_global[np.ix_(free_dofs, free_dofs)]
    P_f = P_global[free_dofs]
    try:
        u_f = scipy.linalg.solve(K_ff, P_f, assume_a='sym')
    except scipy.linalg.LinAlgError:
        raise ValueError('Singular stiffness matrix after applying boundary conditions')
    u_global = np.zeros(n_dofs)
    u_global[free_dofs] = u_f
    K_g_global = np.zeros((n_dofs, n_dofs))
    for elem in elements:
        i = elem['node_i']
        j = elem['node_j']
        coords_i = node_coords[i]
        coords_j = node_coords[j]
        L = np.linalg.norm(coords_j - coords_i)
        vx = (coords_j - coords_i) / L
        if elem.get('local_z') is not None:
            vz = np.array(elem['local_z'])
            vz = vz / np.linalg.norm(vz)
            vy = np.cross(vz, vx)
            vy = vy / np.linalg.norm(vy)
            vz = np.cross(vx, vy)
        else:
            if abs(vx[2]) > 0.9:
                vy = np.array([0.0, 1.0, 0.0])
            else:
                vy = np.array([0.0, 0.0, 1.0])
            vy = vy - np.dot(vy, vx) * vx
            vy = vy / np.linalg.norm(vy)
            vz = np.cross(vx, vy)
        R = np.column_stack([vx, vy, vz])
        T = np.zeros((12, 12))
        for k in range(4):
            T[3 * k:3 * k + 3, 3 * k:3 * k + 3] = R
        elem_dofs_global = [6 * i, 6 * i + 1, 6 * i + 2, 6 * i + 3, 6 * i + 4, 6 * i + 5, 6 * j, 6 * j + 1, 6 * j + 2, 6 * j + 3, 6 * j + 4, 6 * j + 5]
        u_elem_global = u_global[elem_dofs_global]
        u_elem_local = T @ u_elem_global
        k_local = local_elastic_stiffness_matrix_3D_beam(elem['E'], elem['nu'], elem['A'], L, elem['Iy'], elem['Iz'], elem['J'])
        f_local = k_local @ u_elem_local
        Fx2 = f_local[6]
        Mx2 = f_local[9]
        My1 = f_local[4]
        Mz1 = f_local[5]
        My2 = f_local[10]
        Mz2 = f_local[11]
        k_g_local = local_geometric_stiffness_matrix_3D_beam(L, elem['A'], elem['I_rho'], Fx2, Mx2, My1, Mz1, My2, Mz2)
        k_g_global = T.T @ k_g_local @ T
        dofs = [6 * i, 6 * i + 1, 6 * i + 2, 6 * i + 3, 6 * i + 4, 6 * i + 5, 6 * j, 6 * j + 1, 6 * j + 2, 6 * j + 3, 6 * j + 4, 6 * j + 5]
        for (idx_i, dof_i) in enumerate(dofs):
            for (idx_j, dof_j) in enumerate(dofs):
                K_g_global[dof_i, dof_j] += k_g_global[idx_i, idx_j]
    K_ff = K_global[np.ix_(free_dofs, free_dofs)]
    K_g_ff = K_g_global[np.ix_(free_dofs, free_dofs)]
    try:
        (eigenvalues, eigenvectors) = scipy.linalg.eig(K_ff, -K_g_ff)
    except scipy.linalg.LinAlgError:
        raise ValueError('Failed to solve generalized eigenvalue problem')
    eigenvalues = np.real(eigenvalues)
    eigenvectors = np.real(eigenvectors)
    positive_eigenvalues = eigenvalues[eigenvalues > 0]
    if len(positive_eigenvalues) == 0:
        raise ValueError('No positive eigenvalues found')
    min_positive_idx = np.argmin(positive_eigenvalues)
    elastic_critical_load_factor = positive_eigenvalues[min_positive_idx]
    eigenvector_f = eigenvectors[:, np.where(eigenvalues == elastic_critical_load_factor)[0][0]]
    deformed_shape_vector = np.zeros(n_dofs)
    deformed_shape_vector[free_dofs] = eigenvector_f
    return (elastic_critical_load_factor, deformed_shape_vector)