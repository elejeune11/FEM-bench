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
    coords = np.asarray(node_coords, dtype=float)
    if coords.ndim != 2 or coords.shape[1] != 3:
        raise ValueError('node_coords must be an array of shape (n_nodes, 3).')
    n_nodes = coords.shape[0]
    K = np.zeros((6 * n_nodes, 6 * n_nodes), dtype=float)
    for elem in elements:
        if 'node_i' not in elem or 'node_j' not in elem:
            raise KeyError("Each element must specify 'node_i' and 'node_j'.")
        ni = int(elem['node_i'])
        nj = int(elem['node_j'])
        if ni < 0 or nj < 0 or ni >= n_nodes or (nj >= n_nodes):
            raise IndexError('Element node indices out of range.')
        xi, yi, zi = coords[ni]
        xj, yj, zj = coords[nj]
        if 'E' not in elem or 'nu' not in elem or 'A' not in elem or ('J' not in elem):
            raise KeyError("Element must include 'E', 'nu', 'A', and 'J'.")
        E = float(elem['E'])
        nu = float(elem['nu'])
        A = float(elem['A'])
        J = float(elem['J'])
        if 'I_y' in elem:
            Iy = float(elem['I_y'])
        elif 'Iy' in elem:
            Iy = float(elem['Iy'])
        else:
            raise KeyError("Element must include 'I_y' (or 'Iy').")
        if 'I_z' in elem:
            Iz = float(elem['I_z'])
        elif 'Iz' in elem:
            Iz = float(elem['Iz'])
        else:
            raise KeyError("Element must include 'I_z' (or 'Iz').")
        dx, dy, dz = (xj - xi, yj - yi, zj - zi)
        L = float(np.sqrt(dx * dx + dy * dy + dz * dz))
        if np.isclose(L, 0.0):
            raise ValueError('Element has zero length.')
        K_local = local_elastic_stiffness_matrix_3D_beam(E, nu, A, L, Iy, Iz, J)
        ref_vec = elem.get('local_z', None)
        Gamma = beam_transformation_matrix_3D(xi, yi, zi, xj, yj, zj, ref_vec)
        K_global_elem = Gamma.T @ K_local @ Gamma
        dofs_i = np.arange(6 * ni, 6 * ni + 6)
        dofs_j = np.arange(6 * nj, 6 * nj + 6)
        dofs = np.concatenate((dofs_i, dofs_j))
        K[np.ix_(dofs, dofs)] += K_global_elem
    return K