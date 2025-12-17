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
        raise ValueError('node_coords must have shape (n_nodes, 3).')
    n_nodes = node_coords.shape[0]
    ndofs = 6 * n_nodes
    K = np.zeros((ndofs, ndofs), dtype=float)
    if not elements:
        return K
    raw_indices = []
    for elem in elements:
        if 'node_i' not in elem or 'node_j' not in elem:
            raise KeyError("Each element must contain 'node_i' and 'node_j'.")
        raw_indices.append(int(elem['node_i']))
        raw_indices.append(int(elem['node_j']))
    is_zero_based = all((idx >= 0 and idx < n_nodes for idx in raw_indices))
    is_one_based = all((idx >= 1 and idx <= n_nodes for idx in raw_indices))
    if is_zero_based:
        base = 0
    elif is_one_based:
        base = 1
    else:
        raise IndexError('Element node indices are out of bounds for the provided node_coords.')
    for elem in elements:
        ni = int(elem['node_i']) - base
        nj = int(elem['node_j']) - base
        xi, yi, zi = node_coords[ni]
        xj, yj, zj = node_coords[nj]
        dx, dy, dz = (xj - xi, yj - yi, zj - zi)
        L = float(np.sqrt(dx * dx + dy * dy + dz * dz))
        try:
            E = float(elem['E'])
            nu = float(elem['nu'])
            A = float(elem['A'])
            Iy = float(elem['I_y'])
            Iz = float(elem['I_z'])
            J = float(elem['J'])
        except KeyError as exc:
            raise KeyError("Element is missing one of the required properties: 'E', 'nu', 'A', 'I_y', 'I_z', 'J'.") from exc
        ref_vec = None
        if 'local_z' in elem and elem['local_z'] is not None:
            v = np.asarray(elem['local_z'], dtype=float)
            if v.ndim != 1 or v.size != 3:
                raise ValueError('local_z must be a length-3 vector.')
            norm_v = float(np.linalg.norm(v))
            if np.isclose(norm_v, 0.0):
                raise ValueError('local_z vector must be non-zero.')
            ref_vec = v / norm_v
        Gamma = beam_transformation_matrix_3D(xi, yi, zi, xj, yj, zj, ref_vec)
        k_local = local_elastic_stiffness_matrix_3D_beam(E, nu, A, L, Iy, Iz, J)
        k_global = Gamma.T @ k_local @ Gamma
        dofs_i = np.arange(6 * ni, 6 * ni + 6)
        dofs_j = np.arange(6 * nj, 6 * nj + 6)
        dofs = np.concatenate((dofs_i, dofs_j))
        K[np.ix_(dofs, dofs)] += k_global
    return K