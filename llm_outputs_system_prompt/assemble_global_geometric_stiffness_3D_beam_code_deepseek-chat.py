def assemble_global_geometric_stiffness_3D_beam(node_coords: np.ndarray, elements: Sequence[dict], u_global: np.ndarray) -> np.ndarray:
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
    n_nodes = node_coords.shape[0]
    n_dofs = 6 * n_nodes
    K_g_global = np.zeros((n_dofs, n_dofs))
    for element in elements:
        i = element['node_i']
        j = element['node_j']
        (xi, yi, zi) = node_coords[i]
        (xj, yj, zj) = node_coords[j]
        local_z = element.get('local_z')
        if local_z is not None:
            local_z = np.array(local_z)
        Gamma = beam_transformation_matrix_3D(xi, yi, zi, xj, yj, zj, local_z)
        u_e_global = np.zeros(12)
        u_e_global[0:6] = u_global[6 * i:6 * i + 6]
        u_e_global[6:12] = u_global[6 * j:6 * j + 6]
        local_loads = compute_local_element_loads_beam_3D(element, xi, yi, zi, xj, yj, zj, u_e_global)
        Fx2 = local_loads[6]
        Mx2 = local_loads[9]
        My1 = local_loads[4]
        Mz1 = local_loads[5]
        My2 = local_loads[10]
        Mz2 = local_loads[11]
        dx = xj - xi
        dy = yj - yi
        dz = zj - zi
        L = np.sqrt(dx * dx + dy * dy + dz * dz)
        A = element['A']
        I_rho = element['I_rho']
        k_g_local = local_geometric_stiffness_matrix_3D_beam(L, A, I_rho, Fx2, Mx2, My1, Mz1, My2, Mz2)
        k_g_global_element = Gamma.T @ k_g_local @ Gamma
        dof_indices = []
        for node in [i, j]:
            dof_indices.extend(range(6 * node, 6 * node + 6))
        for (idx_i, global_i) in enumerate(dof_indices):
            for (idx_j, global_j) in enumerate(dof_indices):
                K_g_global[global_i, global_j] += k_g_global_element[idx_i, idx_j]
    return K_g_global