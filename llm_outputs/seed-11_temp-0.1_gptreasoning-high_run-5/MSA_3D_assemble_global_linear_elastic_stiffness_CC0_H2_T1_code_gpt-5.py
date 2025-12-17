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
    node_coords = np.asarray(node_coords, dtype=float)
    if node_coords.ndim != 2 or node_coords.shape[1] != 3:
        raise ValueError('node_coords must be an array of shape (n_nodes, 3).')
    n_nodes = node_coords.shape[0]
    ndof = 6 * n_nodes
    K = np.zeros((ndof, ndof), dtype=float)
    if elements is None or len(elements) == 0:
        return K
    idx_values = []
    for el in elements:
        if 'node_i' not in el or 'node_j' not in el:
            raise KeyError("Each element must contain 'node_i' and 'node_j'.")
        idx_values.extend([int(el['node_i']), int(el['node_j'])])
    min_idx = int(min(idx_values))
    max_idx = int(max(idx_values))
    if 0 <= min_idx and max_idx <= n_nodes - 1:
        base_offset = 0
    elif 1 <= min_idx and max_idx <= n_nodes:
        base_offset = -1
    else:
        raise IndexError('Element node indices out of bounds for provided node_coords.')
    for el in elements:
        i_raw = int(el['node_i'])
        j_raw = int(el['node_j'])
        i = i_raw + base_offset
        j = j_raw + base_offset
        if not (0 <= i < n_nodes and 0 <= j < n_nodes):
            raise IndexError('Element references invalid node indices after base detection.')
        xi, yi, zi = node_coords[i]
        xj, yj, zj = node_coords[j]
        dx, dy, dz = (xj - xi, yj - yi, zj - zi)
        L = float(np.sqrt(dx * dx + dy * dy + dz * dz))
        try:
            E = float(el['E'])
            nu = float(el['nu'])
            A = float(el['A'])
            Iy = float(el['I_y'])
            Iz = float(el['I_z'])
            J = float(el['J'])
        except KeyError as e:
            raise KeyError("Element must contain 'E', 'nu', 'A', 'I_y', 'I_z', and 'J'.") from e
        ref_vec = el.get('local_z', None)
        if ref_vec is not None:
            ref_vec = np.asarray(ref_vec, dtype=float)
        k_loc = local_elastic_stiffness_matrix_3D_beam(E, nu, A, L, Iy, Iz, J)
        Gamma = beam_transformation_matrix_3D(xi, yi, zi, xj, yj, zj, ref_vec)
        k_glob_e = Gamma.T @ k_loc @ Gamma
        dof_i = np.arange(6 * i, 6 * i + 6, dtype=int)
        dof_j = np.arange(6 * j, 6 * j + 6, dtype=int)
        edofs = np.concatenate((dof_i, dof_j))
        K[np.ix_(edofs, edofs)] += k_glob_e
    return K