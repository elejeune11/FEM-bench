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
        raise ValueError('node_coords must have shape (n_nodes, 3).')
    n_nodes = node_coords.shape[0]
    u_global = np.asarray(u_global, dtype=float)
    ndof_total = 6 * n_nodes
    if u_global.shape != (ndof_total,):
        raise ValueError('u_global must be a 1D array of length 6*n_nodes.')
    K = np.zeros((ndof_total, ndof_total), dtype=float)
    for ele in elements:
        ni = int(ele['node_i'])
        nj = int(ele['node_j'])
        if not (0 <= ni < n_nodes and 0 <= nj < n_nodes):
            raise IndexError('Element node index out of range.')
        xi, yi, zi = node_coords[ni]
        xj, yj, zj = node_coords[nj]
        dx, dy, dz = (xj - xi, yj - yi, zj - zi)
        L = float(np.sqrt(dx * dx + dy * dy + dz * dz))
        if np.isclose(L, 0.0):
            raise ValueError('Element length is zero.')
        dofs = np.concatenate([np.arange(6 * ni, 6 * ni + 6), np.arange(6 * nj, 6 * nj + 6)])
        u_e_global = u_global[dofs]
        required_props = ('E', 'nu', 'A', 'I_y', 'I_z', 'J')
        missing = [k for k in required_props if k not in ele]
        if missing:
            raise KeyError(f'Element missing required properties for internal load computation: {missing}')
        ele_info = {k: ele[k] for k in required_props}
        if 'local_z' in ele:
            ele_info['local_z'] = ele['local_z']
        q_local = compute_local_element_loads_beam_3D(ele_info, xi, yi, zi, xj, yj, zj, u_e_global)
        Fx2 = float(q_local[6])
        Mx2 = float(q_local[9])
        My1 = float(q_local[4])
        Mz1 = float(q_local[5])
        My2 = float(q_local[10])
        Mz2 = float(q_local[11])
        if 'A' not in ele or 'I_rho' not in ele:
            raise KeyError("Element must provide 'A' and 'I_rho' for geometric stiffness.")
        A = float(ele['A'])
        I_rho = float(ele['I_rho'])
        k_g_local = local_geometric_stiffness_matrix_3D_beam(L, A, I_rho, Fx2, Mx2, My1, Mz1, My2, Mz2)
        Gamma = beam_transformation_matrix_3D(xi, yi, zi, xj, yj, zj, ele.get('local_z'))
        k_g_global = Gamma.T @ k_g_local @ Gamma
        K[np.ix_(dofs, dofs)] += k_g_global
    return K