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
        raise ValueError('node_coords must be a 2D array with shape (n_nodes, 3).')
    n_nodes = node_coords.shape[0]
    ndofs = 6 * n_nodes
    K = np.zeros((ndofs, ndofs), dtype=float)
    if elements is None:
        return K
    if not hasattr(elements, '__iter__'):
        raise ValueError('elements must be an iterable of dictionaries.')
    idxs = []
    for el in elements:
        if not isinstance(el, dict):
            raise ValueError('Each element must be a dictionary.')
        if 'node_i' not in el or 'node_j' not in el:
            raise ValueError("Each element dictionary must contain 'node_i' and 'node_j'.")
        try:
            idxs.extend([int(el['node_i']), int(el['node_j'])])
        except Exception:
            raise ValueError('Element node indices must be integers.')
    if len(idxs) == 0:
        return K
    idxs_arr = np.asarray(idxs, dtype=int)
    all_in_1_based = np.all((idxs_arr >= 1) & (idxs_arr <= n_nodes)) and (not np.any(idxs_arr == 0))
    all_in_0_based = np.all((idxs_arr >= 0) & (idxs_arr < n_nodes))
    if all_in_1_based and (not all_in_0_based):
        one_based = True
    elif all_in_0_based:
        one_based = False
    else:
        raise IndexError('Element node indices are out of range for provided node_coords.')
    for el in elements:
        ni_raw = int(el['node_i'])
        nj_raw = int(el['node_j'])
        ni = ni_raw - 1 if one_based else ni_raw
        nj = nj_raw - 1 if one_based else nj_raw
        if ni < 0 or nj < 0 or ni >= n_nodes or (nj >= n_nodes):
            raise IndexError('Element node indices out of bounds.')
        if ni == nj:
            raise ValueError('Element cannot connect a node to itself.')
        xi, yi, zi = node_coords[ni]
        xj, yj, zj = node_coords[nj]
        dx, dy, dz = (xj - xi, yj - yi, zj - zi)
        L = float(np.sqrt(dx * dx + dy * dy + dz * dz))
        if np.isclose(L, 0.0):
            raise ValueError('Element has zero length.')
        for key in ('E', 'nu', 'A', 'I_y', 'I_z', 'J'):
            if key not in el:
                raise ValueError(f"Element missing required property '{key}'.")
        E = float(el['E'])
        nu = float(el['nu'])
        A = float(el['A'])
        Iy = float(el['I_y'])
        Iz = float(el['I_z'])
        J = float(el['J'])
        if not (E > 0.0 and A > 0.0 and (Iy > 0.0) and (Iz > 0.0) and (J > 0.0)):
            raise ValueError('Element properties E, A, I_y, I_z, and J must be positive.')
        if np.isclose(1.0 + nu, 0.0):
            raise ValueError("Invalid Poisson's ratio: 1 + nu must not be zero.")
        local_z = el.get('local_z', None)
        ref_vec = None if local_z is None else np.asarray(local_z, dtype=float)
        Ke_local = local_elastic_stiffness_matrix_3D_beam(E, nu, A, L, Iy, Iz, J)
        Gamma = beam_transformation_matrix_3D(xi, yi, zi, xj, yj, zj, ref_vec)
        Ke_global = Gamma.T @ Ke_local @ Gamma
        dofs_i = np.arange(6 * ni, 6 * ni + 6, dtype=int)
        dofs_j = np.arange(6 * nj, 6 * nj + 6, dtype=int)
        edofs = np.concatenate((dofs_i, dofs_j))
        K[np.ix_(edofs, edofs)] += Ke_global
    return K