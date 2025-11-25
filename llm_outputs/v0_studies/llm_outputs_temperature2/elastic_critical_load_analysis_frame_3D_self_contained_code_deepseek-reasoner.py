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
    n_dofs = 6 * n_nodes
    constrained_dofs = set()
    for (node, bc) in boundary_conditions.items():
        base_dof = 6 * node
        if isinstance(bc[0], bool):
            for (i, is_constrained) in enumerate(bc):
                if is_constrained:
                    constrained_dofs.add(base_dof + i)
        else:
            for dof in bc:
                constrained_dofs.add(base_dof + dof)
    free_dofs = sorted(set(range(n_dofs)) - constrained_dofs)
    if len(free_dofs) == 0:
        raise ValueError('All DOFs are constrained - no free DOFs for analysis')
    K = np.zeros((n_dofs, n_dofs))
    for element in elements:
        node_i = element['node_i']
        node_j = element['node_j']
        E = element['E']
        nu = element['nu']
        A = element['A']
        Iy = element['Iy']
        Iz = element['Iz']
        J = element['J']
        I_rho = element['I_rho']
        local_z = element.get('local_z', None)
        (xi, yi, zi) = node_coords[node_i]
        (xj, yj, zj) = node_coords[node_j]
        L = np.sqrt((xj - xi) ** 2 + (yj - yi) ** 2 + (zj - zi) ** 2)
        cx = (xj - xi) / L
        cy = (yj - yi) / L
        cz = (zj - zi) / L
        if local_z is None:
            if abs(cx) < 0.8 or abs(cz) < 0.8:
                local_z = np.array([0, 0, 1])
            else:
                local_z = np.array([0, 1, 0])
        else:
            local_z = np.array(local_z)
            local_z = local_z / np.linalg.norm(local_z)
        T = np.zeros((12, 12))
        y_axis = np.cross(local_z, [cx, cy, cz])
        y_axis = y_axis / np.linalg.norm(y_axis)
        z_axis = np.cross([cx, cy, cz], y_axis)
        z_axis = z_axis / np.linalg.norm(z_axis)
        R = np.array([[cx, cy, cz], [y_axis[0], y_axis[1], y_axis[2]], [z_axis[0], z_axis[1], z_axis[2]]])
        for i in range(4):
            block_start = 3 * i
            T[block_start:block_start + 3, block_start:block_start + 3] = R
        k_local = np.zeros((12, 12))
        EA_L = E * A / L
        k_local[0, 0] = EA_L
        k_local[0, 6] = -EA_L
        k_local[6, 0] = -EA_L
        k_local[6, 6] = EA_L
        GJ_L = E / (2 * (1 + nu)) * J / L
        k_local[3, 3] = GJ_L
        k_local[3, 9] = -GJ_L
        k_local[9, 3] = -GJ_L
        k_local[9, 9] = GJ_L
        EIy_L3 = E * Iy / L ** 3
        k_local[1, 1] = 12 * EIy_L3
        k_local[1, 5] = 6 * L * EIy_L3
        k_local[1, 7] = -12 * EIy_L3
        k_local[1, 11] = 6 * L * EIy_L3
        k_local[5, 1] = 6 * L * EIy_L3
        k_local[5, 5] = 4 * L ** 2 * EIy_L3
        k_local[5, 7] = -6 * L * EIy_L3
        k_local[5, 11] = 2 * L ** 2 * EIy_L3
        k_local[7, 1] = -12 * EIy_L3
        k_local[7, 5] = -6 * L * EIy_L3
        k_local[7, 7] = 12 * EIy_L3
        k_local[7, 11] = -6 * L * EIy_L3
        k_local[11, 1] = 6 * L * EIy_L3
        k_local[11, 5] = 2 * L ** 2 * EIy_L3
        k_local[11, 7] = -6 * L * EIy_L3
        k_local[11, 11] = 4 * L ** 2 * EIy_L3
        EIz_L3 = E * Iz / L ** 3
        k_local[2, 2] = 12 * EIz_L3
        k_local[2, 4] = -6 * L * EIz_L3
        k_local[2, 8] = -12 * EIz_L3
        k_local[2, 10] = -6 * L * EIz_L3
        k_local[4, 2] = -6 * L * EIz_L3
        k_local[4, 4] = 4 * L ** 2 * EIz_L3
        k_local[4, 8] = 6 * L * EIz_L3
        k_local[4, 10] = 2 * L ** 2 * EIz_L3
        k_local[8, 2] = -12 * EIz_L3
        k_local[8, 4] = 6 * L * EIz_L3
        k_local[8, 8] = 12 * EIz_L3
        k_local[8, 10] = 6 * L * EIz_L3
        k_local[10, 2] = -6 * L * EIz_L3
        k_local[10, 4] = 2 * L ** 2 * EIz_L3
        k_local[10, 8] = 6 * L * EIz_L3
        k_local[10, 10] = 4 * L ** 2 * EIz_L3
        k_global = T.T @ k_local @ T
        dof_indices = []
        for node in [node_i, node_j]:
            base_dof = 6 * node
            dof_indices.extend(range(base_dof, base_dof + 6))
        for (i, dof_i) in enumerate(dof_indices):
            for (j, dof_j) in enumerate(dof_indices):
                K[dof_i, dof_j] += k_global[i, j]
    P = np.zeros(n_dofs)
    for (node, loads) in nodal_loads.items():
        base_dof = 6 * node
        for i in range(6):
            P[base_dof + i] += loads[i]
    K_free = K[np.ix_(free_dofs, free_dofs)]
    P_free = P[free_dofs]
    try:
        u_free = scipy.linalg.solve(K_free, P_free, assume_a='sym')
    except scipy.linalg.LinAlgError:
        raise ValueError('Singular stiffness matrix - check boundary conditions')
    u_full = np.zeros(n_dofs)
    u_full[free_dofs] = u_free
    K_g = np.zeros((n_dofs, n_dofs))
    for element in elements:
        node_i = element['node_i']
        node_j = element['node_j']
        E = element['E']
        A = element['A']
        I_rho = element['I_rho']
        local_z = element.get('local_z', None)
        (xi, yi, zi) = node_coords[node_i]
        (xj, yj, zj) = node_coords[node_j]
        L = np.sqrt((xj - xi) ** 2 + (yj - yi) ** 2 + (zj - zi) ** 2)
        cx = (xj - xi) / L
        cy = (yj - yi) / L
        cz = (zj - zi) / L
        if local_z is None:
            if abs(cx) < 0.8 or abs(cz) < 0.8:
                local_z = np.array([0, 0, 1])
            else:
                local_z = np.array([0, 1, 0])
        else:
            local_z = np.array(local_z)
            local_z = local_z / np.linalg.norm(local_z)
        y_axis = np.cross(local_z, [cx, cy, cz])
        y_axis = y_axis / np.linalg.norm(y_axis)
        z_axis = np.cross([cx, cy, cz], y_axis)
        z_axis = z_axis / np.linalg.norm(z_axis)
        R = np.array([[cx, cy, cz], [y_axis[0], y_axis[1], y_axis[2]], [z_axis[0], z_axis[1], z_axis[2]]])
        T = np.zeros((12, 12))
        for i in range(4):
            block_start = 3 * i
            T[block_start:block_start + 3, block_start:block_start + 3] = R
        dof_indices = []
        for node in [node_i, node_j]:
            base_dof = 6 * node
            dof_indices.extend(range(base_dof, base_dof + 6))
        u_element = u_full[dof_indices]
        u_element_local = T @ u_element