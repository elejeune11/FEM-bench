def solve_linear_elastic_frame_3D_self_contained(node_coords: np.ndarray, elements: Sequence[dict], boundary_conditions: dict[int, Sequence[int]], nodal_loads: dict[int, Sequence[float]]):
    """
    Small-displacement linear-elastic analysis of a 3D frame using beam elements.
    The function assembles the global stiffness matrix (K) and load vector (P),
    partitions degrees of freedom (DOFs) into free and fixed sets, solves the
    reduced system for displacements at the free DOFs, and computes true support
    reactions at the fixed DOFs.
    Coordinate system: global right-handed Cartesian. Element local axes follow the
    beam axis (local x) with orientation defined via a reference vector.
    Condition number policy: the system is solved only if the free–free stiffness
    submatrix K_ff is well-conditioned (cond(K_ff) < 1e16). Otherwise a ValueError
    is raised.
    Parameters
    ----------
    node_coords : (N, 3) float ndarray
        Global coordinates of the N nodes (row i → [x, y, z] of node i, 0-based).
    elements : iterable of dict
        Each dict must contain:
            'node_i', 'node_j' : int
                End node indices (0-based).
            'E', 'nu', 'A', 'I_y', 'I_z', 'J' : float
                Material and geometric properties.
            'local_z' : (3,) array or None
                Optional unit vector to define the local z-direction for transformation
                matrix orientation (must be unit length and not parallel to the beam axis).
    boundary_conditions : dict[int, Sequence[int]]
        Maps node index → 6-element iterable of 0 (free) or 1 (fixed). Omitted nodes ⇒ all DOFs free.
    nodal_loads : dict[int, Sequence[float]]
        Maps node index → 6-element [Fx, Fy, Fz, Mx, My, Mz]. Omitted nodes ⇒ zero loads.
    Returns
    -------
    u : (6*N,) ndarray
        Global displacement vector ordered as [UX, UY, UZ, RX, RY, RZ] for each node
        in sequence. Values are computed at free DOFs; fixed DOFs are zero.
    r : (6*N,) ndarray
        Global reaction force/moment vector with nonzeros only at fixed DOFs.
        Reactions are computed as internal elastic forces minus applied loads at the
        fixed DOFs; free DOFs have zero entries.
    Raises
    ------
    ValueError
        If the free-free submatrix K_ff is ill-conditioned (cond(K_ff) ≥ 1e16).
    Notes
    -----
    """
    n_nodes = node_coords.shape[0]
    n_dofs = 6 * n_nodes
    K_global = np.zeros((n_dofs, n_dofs))
    P_global = np.zeros(n_dofs)
    for (node_idx, loads) in nodal_loads.items():
        dof_start = 6 * node_idx
        P_global[dof_start:dof_start + 6] = loads
    for elem in elements:
        i = elem['node_i']
        j = elem['node_j']
        E = elem['E']
        nu = elem['nu']
        A = elem['A']
        I_y = elem['I_y']
        I_z = elem['I_z']
        J = elem['J']
        local_z_ref = elem.get('local_z', None)
        (xi, yi, zi) = node_coords[i]
        (xj, yj, zj) = node_coords[j]
        dx = xj - xi
        dy = yj - yi
        dz = zj - zi
        L = np.sqrt(dx ** 2 + dy ** 2 + dz ** 2)
        cx = dx / L
        cy = dy / L
        cz = dz / L
        if local_z_ref is not None:
            z_ref = np.array(local_z_ref)
            z_ref = z_ref / np.linalg.norm(z_ref)
            if abs(np.dot([cx, cy, cz], z_ref)) > 0.98:
                raise ValueError('local_z vector is nearly parallel to beam axis')
            ly = np.cross(z_ref, [cx, cy, cz])
            ly = ly / np.linalg.norm(ly)
            lz = np.cross([cx, cy, cz], ly)
            lz = lz / np.linalg.norm(lz)
        elif abs(cx) < 0.98 and abs(cy) < 0.98:
            ly = np.array([-cy, cx, 0])
            ly = ly / np.linalg.norm(ly)
            lz = np.cross([cx, cy, cz], ly)
            lz = lz / np.linalg.norm(lz)
        else:
            ly = np.array([0, 1, 0])
            lz = np.cross([cx, cy, cz], ly)
            lz = lz / np.linalg.norm(lz)
        T_3x3 = np.column_stack([[cx, cy, cz], ly, lz])
        T_12x12 = np.zeros((12, 12))
        for block in range(4):
            start_row = 3 * block
            start_col = 3 * block
            T_12x12[start_row:start_row + 3, start_col:start_col + 3] = T_3x3
        k_local = np.zeros((12, 12))
        G = E / (2 * (1 + nu))
        k_axial = E * A / L
        k_local[0, 0] = k_axial
        k_local[0, 6] = -k_axial
        k_local[6, 0] = -k_axial
        k_local[6, 6] = k_axial
        k_torsion = G * J / L
        k_local[3, 3] = k_torsion
        k_local[3, 9] = -k_torsion
        k_local[9, 3] = -k_torsion
        k_local[9, 9] = k_torsion
        phi_y = 12 * E * I_y / L ** 3
        psi_y = 6 * E * I_y / L ** 2
        chi_y = 4 * E * I_y / L
        omega_y = 2 * E * I_y / L
        k_local[2, 2] = phi_y
        k_local[2, 4] = psi_y
        k_local[2, 8] = -phi_y
        k_local[2, 10] = psi_y
        k_local[4, 2] = psi_y
        k_local[4, 4] = chi_y
        k_local[4, 8] = -psi_y
        k_local[4, 10] = omega_y
        k_local[8, 2] = -phi_y
        k_local[8, 4] = -psi_y
        k_local[8, 8] = phi_y
        k_local[8, 10] = -psi_y
        k_local[10, 2] = psi_y
        k_local[10, 4] = omega_y
        k_local[10, 8] = -psi_y
        k_local[10, 10] = chi_y
        phi_z = 12 * E * I_z / L ** 3
        psi_z = 6 * E * I_z / L ** 2
        chi_z = 4 * E * I_z / L
        omega_z = 2 * E * I_z / L
        k_local[1, 1] = phi_z
        k_local[1, 5] = -psi_z
        k_local[1, 7] = -phi_z
        k_local[1, 11] = -psi_z
        k_local[5, 1] = -psi_z
        k_local[5, 5] = chi_z
        k_local[5, 7] = psi_z
        k_local[5, 11] = omega_z
        k_local[7, 1] = -phi_z
        k_local[7, 5] = psi_z
        k_local[7, 7] = phi_z
        k_local[7, 11] = psi_z
        k_local[11, 1] = -psi_z
        k_local[11, 5] = omega_z
        k_local[11, 7] = psi_z
        k_local[11, 11] = chi_z
        k_global_elem = T_12x12.T @ k_local @ T_12x12
        dof_i = 6 * i
        dof_j = 6 * j
        K_global[dof_i:dof_i + 6, dof_i:dof_i + 6] += k_global_elem[0:6, 0:6]
        K_global[dof_i:dof_i + 6, dof_j:dof_j + 6] += k_global_elem[0:6, 6:12]
        K_global[dof_j:dof_j + 6, dof_i:dof_i + 6] += k_global_elem[6:12, 0:6]
        K_global[dof_j:dof_j + 6, dof_j:dof_j + 6] += k_global_elem[6:12, 6:12]
    fixed_dofs = []
    for (node_idx, bcs) in boundary_conditions.items():
        dof_start = 6 * node_idx
        for (i, fixed) in enumerate(bcs):
            if fixed:
                fixed_dofs.append(dof_start + i)
    free_dofs = [i for i in range(n_dofs) if i not in fixed_dofs]
    K_ff = K_global[np.ix_(free_dofs, free_dofs)]
    P_f = P_global[free_dofs]
    cond_num = np.linalg.cond(K_ff)
    if cond_num >= 1e+16:
        raise ValueError(f'Free-free stiffness matrix is ill-conditioned (cond = {cond_num:.2e})')
    u_f = np.linalg.solve(K_ff, P_f)
    u = np.zeros(n_dofs)
    u[free_dofs] = u_f
    r_full = K_global @ u - P_global
    r = np.zeros(n_dofs)
    r[fixed_dofs] = r_full[fixed_dofs]
    return (u, r)