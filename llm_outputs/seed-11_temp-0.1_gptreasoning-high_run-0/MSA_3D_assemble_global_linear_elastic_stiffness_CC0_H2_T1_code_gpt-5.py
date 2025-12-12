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
    import numpy as np
    node_coords = np.asarray(node_coords, dtype=float)
    if node_coords.ndim != 2 or node_coords.shape[1] != 3:
        raise ValueError('node_coords must be an array of shape (n_nodes, 3).')
    n_nodes = node_coords.shape[0]
    ndofs = 6 * n_nodes
    K = np.zeros((ndofs, ndofs), dtype=float)
    if elements is None:
        return K
    for elem in elements:
        if 'node_i' not in elem or 'node_j' not in elem:
            raise KeyError("Each element must contain 'node_i' and 'node_j' keys.")
        raw_i = int(elem['node_i'])
        raw_j = int(elem['node_j'])

        def to_zero_based(idx, n):
            if 0 <= idx < n:
                return idx
            if 1 <= idx <= n:
                return idx - 1
            raise IndexError('Node index out of bounds for the provided node_coords.')
        i = to_zero_based(raw_i, n_nodes)
        j = to_zero_based(raw_j, n_nodes)
        if any((k not in elem for k in ('E', 'nu', 'A', 'I_y', 'I_z', 'J'))):
            raise KeyError("Each element must include 'E', 'nu', 'A', 'I_y', 'I_z', and 'J'.")
        E = float(elem['E'])
        nu = float(elem['nu'])
        A = float(elem['A'])
        Iy = float(elem['I_y'])
        Iz = float(elem['I_z'])
        J = float(elem['J'])
        (xi, yi, zi) = node_coords[i]
        (xj, yj, zj) = node_coords[j]
        (dx, dy, dz) = (xj - xi, yj - yi, zj - zi)
        L = float(np.sqrt(dx * dx + dy * dy + dz * dz))
        if np.isclose(L, 0.0):
            raise ValueError('Element has zero length (coincident nodes).')
        ref_vec = elem.get('local_z', None)
        if ref_vec is not None:
            ref_vec = np.asarray(ref_vec, dtype=float).reshape(-1)
            if ref_vec.size == 3:
                nrm = np.linalg.norm(ref_vec)
                if not np.isclose(nrm, 0.0):
                    ref_vec = ref_vec / nrm
        k_local = local_elastic_stiffness_matrix_3D_beam(E, nu, A, L, Iy, Iz, J)
        Gamma = beam_transformation_matrix_3D(xi, yi, zi, xj, yj, zj, ref_vec)
        k_global = Gamma.T @ k_local @ Gamma
        dofs_i = np.arange(6 * i, 6 * i + 6)
        dofs_j = np.arange(6 * j, 6 * j + 6)
        element_dofs = np.concatenate((dofs_i, dofs_j))
        K[np.ix_(element_dofs, element_dofs)] += k_global
    return K