def MSA_3D_assemble_global_geometric_stiffness_CC1_H4_T1(node_coords: np.ndarray, elements: Sequence[dict], u_global: np.ndarray) -> np.ndarray:
    """
    Assemble the global geometric stiffness matrix K_g for a 3D beam/frames model
    under the current global displacement state.
    This routine loops over 2-node beam elements, forms each element's local
    geometric stiffness `k_g_local` from axial force and end moments computed in
    the local frame, maps it to the global frame via the 12x12 transformation
    `Gamma`, and scatters the result into the global matrix.
    Parameters
    ----------
    node_coords : (n_nodes, 3) ndarray of float
        Cartesian coordinates (x, y, z) for each node, indexed by node id.
    elements : sequence of dict
        Each element dict must provide at least:
                End-node indices (0-based).
                Cross-sectional area.
                Polar (or appropriate torsional) second moment used by 
                `local_geometric_stiffness_matrix_3D_beam`
                Optional approximate local z-axis direction (3,) used to
                disambiguate the element's local triad in the transformation.
        Additional fields may be present.
    u_global : (6*n_nodes,) ndarray of float
        Global displacement DOF vector with 6 DOF per node in this order:
        [u_x, u_y, u_z, θ_x, θ_y, θ_z] for node 0, then node 1, etc.
    Returns
    -------
    K : (6*n_nodes, 6*n_nodes) ndarray of float
        The assembled geometric stiffness matrix.
    External Dependencies
    ---------------------
    beam_transformation_matrix_3D(xi, yi, zi, xj, yj, zj, local_z=None) -> (12,12) ndarray
        Returns the standard 3D beam transformation matrix that maps local
        element DOFs to global DOFs (d_g = Γ^T d_l). Assumes 2-node, 6 DOF/node.
    compute_local_element_loads_beam_3D(ele, xi, yi, zi, xj, yj, zj, u_e_global) -> (12,) ndarray
        Computes the element end forces/moments in the local frame
        corresponding to the current displacement state. This function assumes the returned local load vector 
        is: [Fxi, Fyi, Fzi, Mxi, Myi, Mzi, Fxj, Fyj, Fzj, Mxj, Myj, Mzj]
        i.e., translational forces then moments at node i, then node j.
    local_geometric_stiffness_matrix_3D_beam(L, A, I_rho, Fx2, Mx2, My1, Mz1, My2, Mz2) -> (12,12) ndarray
        Must return the local geometric stiffness using the element length L, section properties, and local end force resultants as shown.
    """
    node_coords = np.asarray(node_coords, dtype=float)
    if node_coords.ndim != 2 or node_coords.shape[1] != 3:
        raise ValueError('node_coords must be an array of shape (n_nodes, 3)')
    n_nodes = node_coords.shape[0]
    ndofs = 6 * n_nodes
    u_global = np.asarray(u_global, dtype=float).reshape(-1)
    if u_global.size != ndofs:
        raise ValueError('u_global must have length 6*n_nodes')
    K = np.zeros((ndofs, ndofs), dtype=float)
    for e_idx, ele in enumerate(elements):
        if 'node_i' not in ele or 'node_j' not in ele:
            raise KeyError("Each element must define 'node_i' and 'node_j'")
        i = int(ele['node_i'])
        j = int(ele['node_j'])
        if i < 0 or i >= n_nodes or j < 0 or (j >= n_nodes):
            raise IndexError('Element node indices out of range')
        if 'A' not in ele or 'I_rho' not in ele:
            raise KeyError("Each element must provide 'A' and 'I_rho'")
        xi, yi, zi = node_coords[i]
        xj, yj, zj = node_coords[j]
        dx, dy, dz = (xj - xi, yj - yi, zj - zi)
        L = float(np.sqrt(dx * dx + dy * dy + dz * dz))
        Gamma = beam_transformation_matrix_3D(xi, yi, zi, xj, yj, zj, ele.get('local_z'))
        dofs_i = np.arange(6 * i, 6 * i + 6)
        dofs_j = np.arange(6 * j, 6 * j + 6)
        edofs = np.concatenate((dofs_i, dofs_j))
        u_e = u_global[edofs]
        if all((k in ele for k in ('Fx2', 'Mx2', 'My1', 'Mz1', 'My2', 'Mz2'))):
            Fx2 = float(ele['Fx2'])
            Mx2 = float(ele['Mx2'])
            My1 = float(ele['My1'])
            Mz1 = float(ele['Mz1'])
            My2 = float(ele['My2'])
            Mz2 = float(ele['Mz2'])
        elif all((k in ele for k in ('E', 'nu', 'A', 'I_y', 'I_z', 'J'))):
            loads_local = compute_local_element_loads_beam_3D(ele, xi, yi, zi, xj, yj, zj, u_e)
            Fx2 = float(loads_local[6])
            Mx2 = float(loads_local[9])
            My1 = float(loads_local[4])
            Mz1 = float(loads_local[5])
            My2 = float(loads_local[10])
            Mz2 = float(loads_local[11])
        else:
            Fx2 = 0.0
            Mx2 = 0.0
            My1 = 0.0
            Mz1 = 0.0
            My2 = 0.0
            Mz2 = 0.0
        k_g_local = local_geometric_stiffness_matrix_3D_beam(L, float(ele['A']), float(ele['I_rho']), Fx2, Mx2, My1, Mz1, My2, Mz2)
        k_g_global = Gamma.T @ k_g_local @ Gamma
        K[np.ix_(edofs, edofs)] += k_g_global
    return K