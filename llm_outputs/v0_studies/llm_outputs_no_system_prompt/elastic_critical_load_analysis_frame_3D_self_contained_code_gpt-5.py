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
    import numpy as np
    import scipy

    def _norm(v: np.ndarray) -> float:
        return float(np.linalg.norm(v))

    def _unit(v: np.ndarray) -> np.ndarray:
        n = _norm(v)
        if n == 0.0:
            raise ValueError('Zero-length vector encountered while constructing local axes.')
        return v / n

    def _rotation_matrix_from_element(xi: np.ndarray, xj: np.ndarray, local_z_hint: Optional[Sequence[float]]) -> np.ndarray:
        ex = _unit(xj - xi)
        if local_z_hint is None:
            vref = np.array([0.0, 0.0, 1.0], dtype=float)
            if abs(np.dot(ex, vref)) > 0.99:
                vref = np.array([0.0, 1.0, 0.0], dtype=float)
        else:
            vref = np.asarray(local_z_hint, dtype=float)
            if _norm(vref) == 0.0 or abs(np.dot(_unit(vref), ex)) > 0.999:
                vref = np.array([0.0, 0.0, 1.0], dtype=float)
                if abs(np.dot(ex, vref)) > 0.99:
                    vref = np.array([0.0, 1.0, 0.0], dtype=float)
        ez_temp = vref - np.dot(vref, ex) * ex
        ez = _unit(ez_temp)
        ey = np.cross(ez, ex)
        R = np.column_stack((ex, ey, ez))
        return R

    def _elastic_stiffness_local(E: float, nu: float, A: float, Iy: float, Iz: float, J: float, L: float) -> np.ndarray:
        G = E / (2.0 * (1.0 + nu))
        Ke = np.zeros((12, 12), dtype=float)
        k_ax = E * A / L
        Ke[0, 0] += k_ax
        Ke[0, 6] -= k_ax
        Ke[6, 0] -= k_ax
        Ke[6, 6] += k_ax
        k_t = G * J / L
        Ke[3, 3] += k_t
        Ke[3, 9] -= k_t
        Ke[9, 3] -= k_t
        Ke[9, 9] += k_t
        k_bz = E * Iz / L ** 3
        kb = k_bz * np.array([[12.0, 6.0 * L, -12.0, 6.0 * L], [6.0 * L, 4.0 * L ** 2, -6.0 * L, 2.0 * L ** 2], [-12.0, -6.0 * L, 12.0, -6.0 * L], [6.0 * L, 2.0 * L ** 2, -6.0 * L, 4.0 * L ** 2]], dtype=float)
        idx_bz = [1, 5, 7, 11]
        for a in range(4):
            for b in range(4):
                Ke[idx_bz[a], idx_bz[b]] += kb[a, b]
        k_by = E * Iy / L ** 3
        kb2 = k_by * np.array([[12.0, 6.0 * L, -12.0, 6.0 * L], [6.0 * L, 4.0 * L ** 2, -6.0 * L, 2.0 * L ** 2], [-12.0, -6.0 * L, 12.0, -6.0 * L], [6.0 * L, 2.0 * L ** 2, -6.0 * L, 4.0 * L ** 2]], dtype=float)
        idx_by = [2, 4, 8, 10]
        for a in range(4):
            for b in range(4):
                Ke[idx_by[a], idx_by[b]] += kb2[a, b]
        return Ke

    def _geometric_stiffness_local(N: float, L: float) -> np.ndarray:
        Kg = np.zeros((12, 12), dtype=float)
        if L <= 0.0:
            return Kg
        coeff = N / (30.0 * L)
        kg4 = coeff * np.array([[36.0, 3.0 * L, -36.0, 3.0 * L], [3.0 * L, 4.0 * L ** 2, -3.0 * L, -1.0 * L ** 2], [-36.0, -3.0 * L, 36.0, -3.0 * L], [3.0 * L, -1.0 * L ** 2, -3.0 * L, 4.0 * L ** 2]], dtype=float)
        idx_bz = [1, 5, 7, 11]
        for a in range(4):
            for b in range(4):
                Kg[idx_bz[a], idx_bz[b]] += kg4[a, b]
        idx_by = [2, 4, 8, 10]
        for a in range(4):
            for b in range(4):
                Kg[idx_by[a], idx_by[b]] += kg4[a, b]
        return Kg
    n_nodes = int(node_coords.shape[0])
    if node_coords.shape[1] != 3:
        raise ValueError('node_coords must have shape (n_nodes, 3).')
    dof_per_node = 6
    ndof = n_nodes * dof_per_node
    K = np.zeros((ndof, ndof), dtype=float)
    P = np.zeros(ndof, dtype=float)
    if nodal_loads is not None:
        for (n, load) in nodal_loads.items():
            if n < 0 or n >= n_nodes:
                raise ValueError(f'Load specified for invalid node index {n}.')
            load_arr = np.asarray(load, dtype=float)
            if load_arr.shape != (6,):
                raise ValueError('Each nodal load must be a sequence of 6 components [Fx, Fy, Fz, Mx, My, Mz].')
            base = n * dof_per_node
            P[base:base + dof_per_node] += load_arr
    for e in elements:
        i = int(e['node_i'])
        j = int(e['node_j'])
        if i < 0 or i >= n_nodes or j < 0 or (j >= n_nodes):
            raise ValueError(f'Element references invalid node indices: {i}, {j}')
        xi = np.asarray(node_coords[i], dtype=float)
        xj = np.asarray(node_coords[j], dtype=float)
        L = _norm(xj - xi)
        if L <= 0.0:
            raise ValueError('Element length must be positive.')
        R = _rotation_matrix_from_element(xi, xj, e.get('local_z', None))
        T = np.zeros((12, 12), dtype=float)
        for blk in range(4):
            T[blk * 3:(blk + 1) * 3, blk * 3:(blk + 1) * 3] = R
        E = float(e['E'])
        nu = float(e['nu'])
        A = float(e['A'])
        Iy = float(e['Iy'])
        Iz = float(e['Iz'])
        J = float(e['J'])
        _ = e.get('I_rho', None)
        Ke_local = _elastic_stiffness_local(E, nu, A, Iy, Iz, J, L)
        Ke_global = T @ Ke_local @ T.T
        dof_map = np.array([i * dof_per_node + 0, i * dof_per_node + 1, i * dof_per_node + 2, i * dof_per_node + 3, i * dof_per_node + 4, i * dof_per_node + 5, j * dof_per_node + 0, j * dof_per_node + 1, j * dof_per_node + 2, j * dof_per_node + 3, j * dof_per_node + 4, j * dof_per_node + 5], dtype=int)
        K[np.ix_(dof_map, dof_map)] += Ke_global
    constrained = np.zeros(ndof, dtype=bool)
    if boundary_conditions is not None:
        for (n, spec) in boundary_conditions.items():
            if n < 0 or n >= n_nodes:
                raise ValueError(f'Boundary condition specified for invalid node index {n}.')
            base = n * dof_per_node
            spec_seq = list(spec)
            is_bool_seq = len(spec_seq) == 6 and all((isinstance(val, (bool, np.bool_)) for val in spec_seq))
            if is_bool_seq:
                for (k, flag) in enumerate(spec_seq):
                    if flag:
                        constrained[base + k] = True
            else:
                try:
                    for idx in spec_seq:
                        idxi = int(idx)
                        if idxi < 0 or idxi >= 6:
                            raise ValueError
                        constrained[base + idxi] = True
                except Exception:
                    raise ValueError('Boundary condition specification must be a sequence of 6 booleans or a sequence of DOF indices (0..5).')
    free = np.where(~constrained)[0]
    if free.size == 0:
        raise ValueError('No free DOFs remain after applying boundary conditions.')
    K_ff = K[np.ix_(free, free)]
    P_f = P[free]
    try:
        u_f = scipy.linalg.solve(K_ff, P_f, assume_a='sym')
    except Exception as ex:
        try:
            u_f = scipy.linalg.solve(K_ff, P_f)
        except Exception:
            raise ValueError(f'Failed to solve the reduced linear system (possible rigid-body modes or singular stiffness): {ex}')
    u = np.zeros(ndof, dtype=float)
    u[free] = u_f
    Kg = np.zeros((ndof, ndof), dtype=float)
    for e in elements:
        i = int(e['node_i'])
        j = int(e['node_j'])
        xi = np.asarray(node_coords[i], dtype=float)
        xj = np.asarray(node_coords[j], dtype=float)
        L = _norm(xj - xi)
        R = _rotation_matrix_from_element(xi, xj, e.get('local_z', None))
        T = np.zeros((12, 12), dtype=float)
        for blk in range(4):
            T[blk * 3:(blk + 1) * 3, blk * 3:(blk + 1) * 3] = R
        E = float(e['E'])
        nu = float(e['nu'])
        A = float(e['A'])
        Iy = float(e['Iy'])
        Iz = float(e['Iz'])
        J = float(e['J'])
        dof_map = np.array([i * dof_per_node + 0, i * dof_per_node + 1, i * dof_per_node + 2, i * dof_per_node + 3, i * dof_per_node + 4, i * dof_per_node + 5, j * dof_per_node + 0, j * dof_per_node + 1, j * dof_per_node + 2, j * dof_per_node + 3, j * dof_per_node + 4, j * dof_per_node + 5], dtype=int)
        ue_g = u[dof_map]
        ue_l = T.T @ ue_g
        uxi = ue_l[0]
        uxj = ue_l[6]
        N = -(E * A / L) * (uxj - uxi)
        Kg_local = _geometric_stiffness_local(N, L)
        Kg_global = T @ Kg_local @ T.T
        Kg[np.ix_(dof_map, dof_map)] += Kg_global
    Kg_ff = Kg[np.ix_(free, free)]
    if np.allclose(Kg_ff, 0.0, atol=1e-12, rtol=1e-12):
        raise ValueError('Geometric stiffness is zero under the provided reference load; no buckling load can be computed.')
    try:
        (eigvals, eigvecs) = scipy.linalg.eig(K_ff, -Kg_ff)
    except Exception as ex:
        raise ValueError(f'Failed to solve the generalized eigenvalue problem: {ex}')
    eigvals = np.asarray(eigvals)
    imag_tol = 1e-08
    pos_tol = 1e-10
    finite_mask = np.isfinite(eigvals.real) & np.isfinite(eigvals.imag)
    real_mask = np.abs(eigvals.imag) <= imag_tol * (1.0 + np.abs(eigvals.real))
    positive_mask = eigvals.real > pos_tol
    mask = finite_mask & real_mask & positive_mask
    if not np.any(mask):
        raise ValueError('No positive real eigenvalue found for the buckling problem.')
    pos_eigs = eigvals.real[mask]
    idx_candidates = np.where(mask)[0]
    min_idx_local = int(np.argmin(pos_eigs))
    eig_index = int(idx_candidates[min_idx_local])
    lambda_cr = float(pos_eigs[min_idx_local])
    phi_f = eigvecs[:, eig_index].real
    phi = np.zeros(ndof, dtype=float)
    phi[free] = phi_f
    return (lambda_cr, phi)