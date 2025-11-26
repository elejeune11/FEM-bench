def MSA_3D_assemble_global_linear_elastic_stiffness_CC0_H2_T1(node_coords, elements):
    """
    Assembles the global stiffness matrix for a 3D linear elastic frame structure composed of beam elements.
    Each beam element connects two nodes and contributes a 12x12 stiffness matrix (6 DOFs per node) to the
    global stiffness matrix. The element stiffness is computed in the local coordinate system using material
    and geometric properties, then transformed into the global coordinate system via a transformation matrix.
    Parameters
    ----------
    node_coords : ndarray of shape (n_nodes, 3)
        Array containing the (x, y, z) coordinates of each node.
    elements : list of dict
        A list of element dictionaries. Each dictionary must contain:
                Indices of the start and end nodes.
                Young's modulus of the element.
                Poisson's ratio of the element.
                Cross-sectional area.
                Second moments of area about local y and z axes.
                Torsional constant.
                Optional vector defining the local z-direction to resolve transformation ambiguity.
    Returns
    -------
    K : ndarray of shape (6 * n_nodes, 6 * n_nodes)
        The assembled global stiffness matrix of the structure, with 6 degrees of freedom per node.
    Helper Functions Required
    -------------------------
        Returns the 12x12 local stiffness matrix for a 3D beam element.
        Returns the 12x12 transformation matrix to convert local stiffness to global coordinates.
    Notes
    -----
    """
    n_nodes = node_coords.shape[0]
    n_dof = 6 * n_nodes
    K = np.zeros((n_dof, n_dof))
    for el in elements:
        idx_i = el['node_i']
        idx_j = el['node_j']
        coords_i = node_coords[idx_i]
        coords_j = node_coords[idx_j]
        (xi, yi, zi) = coords_i
        (xj, yj, zj) = coords_j
        (dx, dy, dz) = (xj - xi, yj - yi, zj - zi)
        L = np.sqrt(dx * dx + dy * dy + dz * dz)
        E = el['E']
        nu = el['nu']
        A = el['A']
        Iy = el['I_y']
        Iz = el['I_z']
        J = el['J']
        k_local = local_elastic_stiffness_matrix_3D_beam(E, nu, A, L, Iy, Iz, J)
        local_z = el.get('local_z')
        ref_vec = None
        if local_z is not None:
            ref_vec = np.array(local_z, dtype=float)
            norm = np.linalg.norm(ref_vec)
            if not np.isclose(norm, 0.0):
                ref_vec = ref_vec / norm
        gamma = beam_transformation_matrix_3D(xi, yi, zi, xj, yj, zj, ref_vec)
        k_global = gamma.T @ k_local @ gamma
        start_i = 6 * idx_i
        end_i = start_i + 6
        start_j = 6 * idx_j
        end_j = start_j + 6
        K[start_i:end_i, start_i:end_i] += k_global[0:6, 0:6]
        K[start_j:end_j, start_j:end_j] += k_global[6:12, 6:12]
        K[start_i:end_i, start_j:end_j] += k_global[0:6, 6:12]
        K[start_j:end_j, start_i:end_i] += k_global[6:12, 0:6]
    return K