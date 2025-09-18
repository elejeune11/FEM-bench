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
    n_dofs_total = 6 * n_nodes
    K_global = np.zeros((n_dofs_total, n_dofs_total))
    P_global = np.zeros(n_dofs_total)
    for (node_idx, load_vec) in nodal_loads.items():
        start_dof = 6 * node_idx
        P_global[start_dof:start_dof + 6] = load_vec
    fixed_dofs = set()
    for (node_idx, bc_spec) in boundary_conditions.items():
        start_dof = 6 * node_idx
        if isinstance(bc_spec[0], bool):
            for (i, is_fixed) in enumerate(bc_spec):
                if is_fixed:
                    fixed_dofs.add(start_dof + i)
        else:
            for dof_offset in bc_spec:
                fixed_dofs.add(start_dof + dof_offset)
    free_dofs = sorted(set(range(n_dofs_total)) - fixed_dofs)
    for elem in elements:
        i = elem['node_i']
        j = elem['node_j']
        E = elem['E']
        nu = elem['nu']
        A = elem['A']
        Iy = elem['Iy']
        Iz = elem['Iz']
        J = elem['J']
        I_rho = elem['I_rho']
        (xi, yi, zi) = node_coords[i]
        (xj, yj, zj) = node_coords[j]
        dx = xj - xi
        dy = yj - yi
        dz = zj - zi
        L = np.sqrt(dx ** 2 + dy ** 2 + dz ** 2)
        k_local = local_elastic_stiffness_matrix_3D_beam(E, nu, A, L, Iy, Iz, J)
        if 'local_z' in elem and elem['local_z'] is not None:
            local_z = np.array(elem['local_z'])
            local_z = local_z / np.linalg.norm(local_z)
            local_x = np.array([dx, dy, dz]) / L
            local_y = np.cross(local_z, local_x)
            local_y = local_y / np.linalg.norm(local_y)
            local_z = np.cross(local_x, local_y)
        else:
            local_x = np.array([dx, dy, dz]) / L
            if abs(local_x[0]) > 1e-06 or abs(local_x[1]) > 1e-06:
                local_z = np.cross(local_x, np.array([0, 0, 1]))
                local_z = local_z / np.linalg.norm(local_z)
            else:
                local_z = np.array([0, 0, 1])
            local_y = np.cross(local_z, local_x)
        T_small = np.vstack([local_x, local_y, local_z])
        T = scipy.linalg.block_diag(T_small, T_small, T_small, T_small)
        k_global_elem = T.T @ k_local @ T
        dof_indices = []
        for node in [i, j]:
            start_dof = 6 * node
            dof_indices