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
    n_nodes = node_coords.shape[0]
    K_g = np.zeros((6 * n_nodes, 6 * n_nodes))
    for ele in elements:
        i = ele['node_i']
        j = ele['node_j']
        (xi, yi, zi) = node_coords[i]
        (xj, yj, zj) = node_coords[j]
        (dx, dy, dz) = (xj - xi, yj - yi, zj - zi)
        L = np.sqrt(dx * dx + dy * dy + dz * dz)
        idx_i = slice(i * 6, i * 6 + 6)
        idx_j = slice(j * 6, j * 6 + 6)
        u_ele_global = np.concatenate((u_global[idx_i], u_global[idx_j]))
        loads_local = compute_local_element_loads_beam_3D(ele, xi, yi, zi, xj, yj, zj, u_ele_global)
        Fx2 = loads_local[6]
        Mx2 = loads_local[9]
        My1 = loads_local[4]
        Mz1 = loads_local[5]
        My2 = loads_local[10]
        Mz2 = loads_local[11]
        A = ele['A']
        I_rho = ele['I_rho']
        k_g_local = local_geometric_stiffness_matrix_3D_beam(L, A, I_rho, Fx2, Mx2, My1, Mz1, My2, Mz2)
        local_z = ele.get('local_z')
        Gamma = beam_transformation_matrix_3D(xi, yi, zi, xj, yj, zj, local_z)
        k_g_ele_global = Gamma.T @ k_g_local @ Gamma
        K_g[idx_i, idx_i] += k_g_ele_global[0:6, 0:6]
        K_g[idx_i, idx_j] += k_g_ele_global[0:6, 6:12]
        K_g[idx_j, idx_i] += k_g_ele_global[6:12, 0:6]
        K_g[idx_j, idx_j] += k_g_ele_global[6:12, 6:12]
    return K_g