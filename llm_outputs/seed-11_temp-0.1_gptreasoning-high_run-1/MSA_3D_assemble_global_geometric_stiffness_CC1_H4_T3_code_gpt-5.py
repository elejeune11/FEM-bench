def MSA_3D_assemble_global_geometric_stiffness_CC1_H4_T3(node_coords: np.ndarray, elements: Sequence[dict], u_global: np.ndarray) -> np.ndarray:
    """
    Assemble the global geometric (initial-stress) stiffness matrix K_g for a 3D frame
    under a given global displacement state.
    Each 2-node Euler–Bernoulli beam contributes a 12×12 local geometric stiffness
    matrix k_g^local that depends on the element length and the internal end
    force/moment resultants induced by the current displacement state. The local
    matrix is then mapped to global coordinates with a 12×12 direction-cosine
    transformation Γ and scattered into the global K_g.
    Parameters
    ----------
    node_coords : (n_nodes, 3) ndarray of float
        Global Cartesian coordinates [x, y, z] of each node (0-based indexing).
    elements : sequence of dict
        Per-element dictionaries. Required keys per element:
            'node_i', 'node_j' : int
                Indices of the start and end nodes.
            'E' : float
                Young's modulus (Pa).
            'nu' : float
                Poisson's ratio (unitless).
            'A' : float
                Cross-sectional area (m²).
            'I_y', 'I_z' : float
                Second moments of area about the local y- and z-axes (m⁴).
            'J' : float
                Torsional constant (m⁴).
            'local_z' : array-like of shape (3,), optional
                Unit vector in global coordinates defining the local z-axis orientation.
                Must not be parallel to the beam axis. If None, a default is chosen (see Notes).
    u_global : (6*n_nodes,) ndarray of float
        Global displacement vector with 6 DOF per node in the order
        [u_x, u_y, u_z, θ_x, θ_y, θ_z] for node 0, then node 1, etc.
    Returns
    -------
    K : (6*n_nodes, 6*n_nodes) ndarray of float
        Assembled global geometric stiffness matrix. For conservative loading and
        the standard formulation, K_g is symmetric.
    Notes
    -----
      unless the beam axis is aligned with global z, in which case use the global y-axis.
      The 'local_z' must be unit length and not parallel to the beam axis.
      induced by the supplied displacement state (not external loads). Their local DOF
      ordering is the same as for local displacements:
      [u1, v1, w1, θx1, θy1, θz1, u2, v2, w2, θx2, θy2, θz2] ↔
      [Fx_i, Fy_i, Fz_i, Mx_i, My_i, Mz_i, Fx_j, Fy_j, Fz_j, Mx_j, My_j, Mz_j].
      should be treated as an error by the transformation routine.
    Effects captured in the geometric stiffness matrix:
        + **Tension (+Fx2)** increases lateral/torsional stiffness.
        + Compression (-Fx2) decreases it and may trigger buckling when K_e + K_g becomes singular.
    """
    node_coords = np.asarray(node_coords, dtype=float)
    if node_coords.ndim != 2 or node_coords.shape[1] != 3:
        raise ValueError("node_coords must be an array of shape (n_nodes, 3)")
    n_nodes = node_coords.shape[0]
    u_global = np.asarray(u_global, dtype=float).reshape(-1)
    if u_global.size != 6 * n_nodes:
        raise ValueError("u_global must have length 6*n_nodes")
    K_global = np.zeros((6 * n_nodes, 6 * n_nodes), dtype=float)
    tol_len = 1e-14
    tol_parallel = 1e-8
    tol_unit = 1e-6
    for e in elements:
        i = int(e['node_i'])
        j = int(e['node_j'])
        if not (0 <= i < n_nodes and 0 <= j < n_nodes):
            raise IndexError("Element node indices out of range")
        if i == j:
            raise ValueError("Element has identical start and end nodes")
        xi = node_coords[i]
        xj = node_coords[j]
        dx = xj - xi
        Le = float(np.linalg.norm(dx))
        if not np.isfinite(Le) or Le <= tol_len:
            raise ValueError("Zero-length or invalid element length")
        ex = dx / Le
        local_z_ref = e.get('local_z', None)
        if local_z_ref is not None:
            zref = np.asarray(local_z_ref, dtype=float).reshape(-1)
            if zref.size != 3:
                raise ValueError("local_z must be array-like of shape (3,)")
            nz = float(np.linalg.norm(zref))
            if not np.isfinite(nz) or nz <= 0.0:
                raise ValueError("local_z has invalid magnitude")
            if abs(nz - 1.0) > tol_unit:
                raise ValueError("local_z must be unit length")
            zref = zref / nz
            if abs(np.dot(zref, ex)) > 1.0 - tol_parallel:
                raise ValueError("local_z must not be parallel to the beam axis")
        else:
            zref = np.array([0.0, 0.0, 1.0], dtype=float)
            if abs(np.dot(zref, ex)) > 1.0 - tol_parallel:
                zref = np.array([0.0, 1.0, 0.0], dtype=float)
        ey_temp = np.cross(zref, ex)
        ny = float(np.linalg.norm(ey_temp))
        if ny <= tol_len:
            raise ValueError("Invalid local_z orientation; nearly parallel to axis")
        ey = ey_temp / ny
        ez = np.cross(ex, ey)
        R = np.vstack((ex, ey, ez))
        Gamma = np.zeros((12, 12), dtype=float)
        for b in range(4):
            Gamma[3 * b:3 * b + 3, 3 * b:3 * b + 3] = R
        dofs_i = [6 * i + k for k in range(6)]
        dofs_j = [6 * j + k for k in range(6)]
        dofs_e = np.array(dofs_i + dofs_j, dtype=int)
        u_e_g = u_global[dofs_e]
        u_e_l = Gamma @ u_e_g
        E = float(e['E'])
        A = float(e['A'])
        N = (E * A / Le) * (u_e_l[0] - u_e_l[6])
        Kg_local = np.zeros((12, 12), dtype=float)
        if Le > tol_len and N != 0.0:
            c = N / (30.0 * Le)
            L = Le  # for readability
            H = np.array([
                [36.0, 3.0 * L, -36.0, 3.0 * L],
                [3.0 * L, 4.0 * L * L, -3.0 * L, -1.0 * L * L],
                [-36.0, -3.0 * L, 36.0, -3.0 * L],
                [3.0 * L, -1.0 * L * L, -3.0 * L, 4.0 * L * L]
            ], dtype=float) * c
            idx_vz = [1, 5, 7, 11]  # v1, θz1, v2, θz2
            for a in range(4):
                for b in range(4):
                    Kg_local[idx_vz[a], idx_vz[b] += H[a, b]
            idx_vy = [2, 4, 8, 10]  # w1, θy1, w2, θy2
            for a in range(4):
                for b in range(4):
                    Kg_local[idx_vy[a], idx_vy[b]] += H[a, b]
        Kg_global_e = Gamma.T @ Kg_local @ Gamma
        for a in range(12):
            ia = dofs_e[a]
            Ka_row = Kg_global_e[a]
            for b in range(12):
                K_global[ia, dofs_e[b]] += Ka_row[b]
    return K_global