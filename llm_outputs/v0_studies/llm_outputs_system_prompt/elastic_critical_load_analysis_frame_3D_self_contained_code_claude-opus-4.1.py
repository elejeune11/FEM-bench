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
        node's specification can be provided in either of two forms:
          the DOF is constrained (fixed).
          at that node are fixed.
        All constrained DOFs are removed from the free set. It is the caller's
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
    n_nodes = node_coords.shape[0]
    n_dof = 6 * n_nodes

    def element_stiffness_3D(node_i_coords, node_j_coords, E, nu, A, Iy, Iz, J, local_z=None):
        dx = node_j_coords[0] - node_i_coords[0]
        dy = node_j_coords[1] - node_i_coords[1]
        dz = node_j_coords[2] - node_i_coords[2]
        L = np.sqrt(dx ** 2 + dy ** 2 + dz ** 2)
        G = E / (2 * (1 + nu))
        k_local = np.zeros((12, 12))
        k_local[0, 0] = k_local[6, 6] = E * A / L
        k_local[0, 6] = k_local[6, 0] = -E * A / L
        k_local[3, 3] = k_local[9, 9] = G * J / L
        k_local[3, 9] = k_local[9, 3] = -G * J / L
        k_local[1, 1] = k_local[7, 7] = 12 * E * Iz / L ** 3
        k_local[1, 5] = k_local[5, 1] = 6 * E * Iz / L ** 2
        k_local[1, 7] = k_local[7, 1] = -12 * E * Iz / L ** 3
        k_local[1, 11] = k_local[11, 1] = 6 * E * Iz / L ** 2
        k_local[5, 5] = 4 * E * Iz / L
        k_local[5, 7] = k_local[7, 5] = -6 * E * Iz / L ** 2
        k_local[5, 11] = k_local[11, 5] = 2 * E * Iz / L
        k_local[7, 11] = k_local[11, 7] = -6 * E * Iz / L ** 2
        k_local[11, 11] = 4 * E * Iz / L
        k_local[2, 2] = k_local[8, 8] = 12 * E * Iy / L ** 3
        k_local[2, 4] = k_local[4, 2] = -6 * E * Iy / L ** 2
        k_local[2, 8] = k_local[8, 2] = -12 * E * Iy / L ** 3
        k_local[2, 10] = k_local[10, 2] = -6 * E * Iy / L ** 2
        k_local[4, 4] = 4 * E * Iy / L
        k_local[4, 8] = k_local[8, 4] = 6 * E * Iy / L ** 2
        k_local[4, 10] = k_local[10, 4] = 2 * E * Iy / L
        k_local[8, 10] = k_local[10, 8] = 6 * E * Iy / L ** 2
        k_local[10, 10] = 4 * E * Iy / L
        e_x = np.array([dx, dy, dz]) / L
        if local_z is not None:
            e_z = np.array(local_z)
            e_z = e_z / np.linalg.norm(e_z)
            e_y = np.cross(e_z, e_x)
            e_y = e_y / np.linalg.norm(e_y)
            e_z = np.cross(e_x, e_y)
        else:
            if abs(e_x[2]) < 0.9:
                ref = np.array([0, 0, 1])
            else:
                ref = np.array([1, 0, 0])
            e_y = np.cross(ref, e_x)
            e_y = e_y / np.linalg.norm(e_y)
            e_z = np.cross(e_x, e_y)
        R = np.array([e_x, e_y, e_z]).T
        T = np.zeros((12, 12))
        for i in range(4):
            T[3 * i:3 * i + 3, 3 * i:3 * i + 3] = R
        return T.T @ k_local @ T

    def element_geometric_stiffness_3D(node_i_coords, node_j_coords, N, My_i, My_j, Mz_i, Mz_j, Mx, Iy, Iz, I_rho, local_z=None):
        dx = node_j_coords[0] - node_i_coords[0]
        dy = node_j_coords[1] - node_i_coords[1]
        dz = node_j_coords[2] - node_i_coords[2]
        L = np.sqrt(dx ** 2 + dy ** 2 + dz ** 2)
        kg_local = np.zeros((12, 12))
        if N != 0:
            kg_local[1, 1] = kg_local[7, 7] = 6 / 5 / L
            kg_local[1, 5] = kg_local[5, 1] = 1 / 10
            kg_local[1, 7] = kg_local[7, 1] = -6 / 5 / L
            kg_local[1, 11] = kg_local[11, 1] = 1 / 10
            kg_local[5, 5] = 2 * L / 15
            kg_local[5, 7] = kg_local[7, 5] = -1 / 10
            kg_local[5, 11] = kg_local[11, 5] = -L / 30
            kg_local[7, 11] = kg_local[11, 7] = -1 / 10
            kg_local[11, 11] = 2 * L / 15
            kg_local[2, 2] = kg_local[8, 8] = 6 / 5 / L
            kg_local[2, 4] = kg_local[4, 2] = -1 / 10
            kg_local[2, 8] = kg_local[8, 2] = -6 / 5 / L
            kg_local[2, 10] = kg_local[10, 2] = -1 / 10
            kg_local[4, 4] = 2 * L / 15
            kg_local[4, 8] = kg_local[8, 4] = 1 / 10
            kg_local[4, 10] = kg_local[10, 4] = -L / 30
            kg_local[8, 10] = kg_local[10, 8] = 1 / 10
            kg_local[10, 10] = 2 * L / 15
            kg_local *= N
        if Mx != 0 and I_rho != 0:
            kg_torsion = np.zeros((12, 12))
            factor = Mx * I_rho / (Iy + Iz) / L
            kg_torsion[1, 4] = kg_torsion[4, 1] = 6 / 5
            kg_torsion[1, 10] = kg_torsion[10, 1] = -6 / 5
            kg_torsion[2, 5] = kg_torsion[5, 2] = -6 / 5
            kg_torsion[2, 11] = kg_torsion[11, 2] = 6 / 5
            kg_torsion[4, 7] = kg_torsion[7, 4] = -6 / 5
            kg_torsion[5, 8] = kg_torsion[8, 5] = 6 / 5
            kg_torsion[7, 10] = kg_torsion[10, 7] = 6 / 5
            kg_torsion[8, 11] = kg_torsion[11, 8] = -6 / 5
            kg_local += factor * kg_torsion
        e_x = np.array([dx, dy, dz]) / L
        if local_z is not None:
            e_z = np.array(local_z)
            e_z = e_z / np.linalg.norm(e_z)
            e_y = np.cross(e_z, e_x)
            e_y = e_y / np.linalg.norm(e_y)
            e_z = np.cross(e_x, e_y)
        else:
            if abs(e_x[2]) < 0.9:
                ref = np.array([0, 0, 1])
            else:
                ref = np.array([1, 0, 0])
            e_y = np.cross(ref, e_x)
            e_y = e_y / np.linalg.norm(e_y)
            e_z = np.cross(e_x, e_y)
        R = np.array([e_x, e_y, e_z]).T
        T = np.zeros((12, 12))
        for i in range(4):
            T[3 * i:3 * i + 3, 3 * i:3 * i + 3] = R
        return T.T @ kg_local @ T
    K = np.zeros((n_dof, n_dof))
    for elem in elements:
        node_i = elem['node_i']
        node_j = elem['node_j']
        k_elem = element_stiffness_3D(node_coords[node_i], node_coords[node_j], elem['E'], elem['nu'], elem['A'], elem['Iy'], elem['Iz'], elem['J'], elem.get('local_z', None))
        dof_i = np.arange(6 * node_i, 6 * node_i + 6)
        dof_j = np.arange(6 * node_j, 6 * node_j + 6)
        dof_elem = np.concatenate([dof_i, dof_j])
        for i in range(12):
            for j in range(12):
                K[dof_elem[i], dof_elem[j]] += k_elem[i, j]
    P = np.zeros(n_dof)
    for (node_idx, loads) in nodal_loads.items():
        dof_start = 6 * node_idx
        P[dof_start:dof_start + 6] = loads
    constrained_dofs = set()
    for (node_idx, bc_spec) in boundary_conditions.items():
        dof_start = 6 * node_idx
        if all((isinstance(x, bool) for x in bc_spec)):
            for (i, is_fixed) in enumerate(bc_spec):
                if is_fixed:
                    constrained_dofs.add(dof_start + i)
        else:
            for local_dof in bc_spec:
                constrained_dofs.add(dof_start + local_dof)
    free_dofs = np.array([i for i in range(n_dof) if i not in constrained_dofs])
    K_ff = K[np.ix_(free_dofs, free_dofs)]
    P_f = P[free_dofs]
    u_f = np.linalg.solve(K_ff, P_f)
    u = np.zeros(n_dof)
    u[free_dofs] = u_f
    K_g = np.zeros((n_dof, n_dof))
    for elem in elements:
        node_i = elem['node_i']
        node_j = elem['node_j']
        k_elem = element_stiffness_3D(node_coords[node_i], node_coords[node_j], elem['E'], elem['nu'], elem['A'], elem['Iy'], elem['Iz'], elem['J'], elem.get('local_z', None))
        dof_i = np.arange(6 * node_i, 6 * node_i + 6)
        dof_j = np.arange(6 * node_j, 6 * node_j + 6)
        dof_elem = np.concatenate([dof_i, dof_j])
        u_elem = u[dof_elem]
        f_elem = k_elem @ u_elem
        dx = node_coords[node_j, 0] - node_coords[node_i, 0]
        dy = node_coords[node_j, 1] - node_coords[node_i, 1]
        dz = node_coords[node_j, 2] - node_coords[node_i, 2]
        L = np.sqrt(dx ** 2 + dy ** 2 + dz ** 2)
        e_x = np.array([dx, dy, dz]) / L
        if elem.get('local_z') is not None:
            e_z = np.array(elem['local_z'])
            e_z = e_z / np.linalg.norm(e_z)
            e_y = np.cross(e_z, e_x)
            e_y = e_y / np.linalg.norm(e_y)
            e_z = np.cross(e_x, e_y)
        else:
            if abs(e_x[2]) < 0.9:
                ref = np.array([0, 0, 1])
            else:
                ref = np.array([1, 0, 0])
            e_y = np.cross(ref, e_x)
            e_y = e_y / np.linalg.norm(e_y)
            e_z = np.cross(e_x, e_y)
        R = np.array([e_x, e_y, e_z]).T
        T = np.zeros((12, 12))
        for i in range(4):
            T[3 * i:3 * i + 3, 3 * i:3 * i + 3] = R
        f_local = T @ f_elem
        N = -f_local[0]
        Mx = -f_local[3]
        My_i = f_local[4]
        My_j = -f_local[10]
        Mz_i = -f_local[5]
        Mz_j = f_local[11]
        kg_elem = element_geometric_stiffness_3D(node_coords[node_i], node_coords[node_j], N, My_i, My_j, Mz_i, Mz_j, Mx, elem['Iy'], elem['Iz'], elem['I_rho'], elem.get('local_z', None))
        for i in range(12):
            for j in range(12):
                K_g[dof_elem[i], dof_elem[j]] += kg_elem[i, j]
    K_ff = K[np.ix_(free_dofs, free_dofs)]
    K_g_ff = K_g[np.ix_(free_dofs, free_dofs)]
    (eigenvalues, eigenvectors) = scipy.linalg.eig(K_ff, -K_g_ff)
    eigenvalues = eigenvalues.real
    eigenvectors = eigenvectors.real
    positive_indices = np.where(eigenvalues > 1e-10)[0]
    if len(positive_indices) == 0:
        raise ValueError('No positive eigenvalues found')
    min_idx = positive_indices[np.argmin(eigenvalues[positive_indices])]
    elastic_critical_load_factor = eigenvalues[min_idx]
    mode_shape = np.zeros(n_dof)
    mode_shape[free_dofs] = eigenvectors[:, min_idx]
    return (elastic_critical_load_factor, mode_shape)