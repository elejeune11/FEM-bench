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
        raise ValueError('node_coords must be of shape (n_nodes, 3).')
    n_nodes = node_coords.shape[0]
    K = np.zeros((6 * n_nodes, 6 * n_nodes), dtype=float)
    for el in elements:
        ni = el['node_i']
        nj = el['node_j']
        if isinstance(ni, (int, np.integer)) and ni > n_nodes - 1 and (1 <= ni <= n_nodes):
            ni = ni - 1
        if isinstance(nj, (int, np.integer)) and nj > n_nodes - 1 and (1 <= nj <= n_nodes):
            nj = nj - 1
        if not (0 <= ni < n_nodes and 0 <= nj < n_nodes):
            raise IndexError('Element node indices out of range.')
        (xi, yi, zi) = node_coords[ni]
        (xj, yj, zj) = node_coords[nj]
        E = float(el['E'])
        nu = float(el['nu'])
        A = float(el['A'])
        Iy = float(el['I_y'])
        Iz = float(el['I_z'])
        J = float(el['J'])
        L = np.linalg.norm(node_coords[nj] - node_coords[ni])
        k_local = local_elastic_stiffness_matrix_3D_beam(E, nu, A, L, Iy, Iz, J)
        ref_vec = None
        if 'local_z' in el and el['local_z'] is not None:
            rv = np.asarray(el['local_z'], dtype=float)
            if rv.shape != (3,):
                raise ValueError('local_z must be length 3 if provided.')
            norm_rv = np.linalg.norm(rv)
            if np.isclose(norm_rv, 0.0):
                raise ValueError('local_z must be non-zero.')
            ref_vec = rv / norm_rv
        Gamma = beam_transformation_matrix_3D(xi, yi, zi, xj, yj, zj, ref_vec)
        k_global = Gamma.T @ k_local @ Gamma
        dof_map = np.concatenate((np.arange(ni * 6, (ni + 1) * 6), np.arange(nj * 6, (nj + 1) * 6)))
        for a in range(12):
            Ia = dof_map[a]
            for b in range(12):
                Ib = dof_map[b]
                K[Ia, Ib] += k_global[a, b]
    return K