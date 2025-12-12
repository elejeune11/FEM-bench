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
    coords = np.asarray(node_coords, dtype=float)
    if coords.ndim != 2 or coords.shape[1] != 3:
        raise ValueError('node_coords must be an array of shape (n_nodes, 3).')
    n_nodes = coords.shape[0]
    ndof = 6 * n_nodes
    K = np.zeros((ndof, ndof), dtype=float)
    for e in elements:
        if 'node_i' not in e or 'node_j' not in e:
            raise KeyError("Each element must have 'node_i' and 'node_j' integer indices.")
        i = int(e['node_i'])
        j = int(e['node_j'])
        if i < 0 or i >= n_nodes or j < 0 or (j >= n_nodes):
            raise IndexError('Element node index out of bounds.')
        (xi, yi, zi) = coords[i]
        (xj, yj, zj) = coords[j]
        (dx, dy, dz) = (xj - xi, yj - yi, zj - zi)
        L = float(np.sqrt(dx * dx + dy * dy + dz * dz))
        if np.isclose(L, 0.0):
            raise ValueError('Element has zero length.')
        required_keys = ['E', 'nu', 'A', 'I_y', 'I_z', 'J']
        if any((k not in e for k in required_keys)):
            raise KeyError("Element must define 'E', 'nu', 'A', 'I_y', 'I_z', 'J'.")
        E = float(e['E'])
        nu = float(e['nu'])
        A = float(e['A'])
        Iy = float(e['I_y'])
        Iz = float(e['I_z'])
        J = float(e['J'])
        k_local = local_elastic_stiffness_matrix_3D_beam(E, nu, A, L, Iy, Iz, J)
        ref = e.get('local_z', None)
        if ref is not None:
            ref = np.asarray(ref, dtype=float)
            if ref.shape != (3,):
                raise ValueError('local_z must be an array-like of length 3.')
            nrm = np.linalg.norm(ref)
            if np.isclose(nrm, 0.0):
                raise ValueError('local_z vector must be non-zero.')
            ref = ref / nrm
        Gamma = beam_transformation_matrix_3D(xi, yi, zi, xj, yj, zj, ref)
        k_global_e = Gamma.T @ k_local @ Gamma
        dofs = np.r_[np.arange(6 * i, 6 * i + 6), np.arange(6 * j, 6 * j + 6)]
        K[np.ix_(dofs, dofs)] += k_global_e
    return K