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
        raise ValueError('node_coords must be of shape (n_nodes, 3)')
    n_nodes = node_coords.shape[0]
    K = np.zeros((6 * n_nodes, 6 * n_nodes), dtype=float)
    for elem in elements:
        if 'node_i' not in elem or 'node_j' not in elem:
            raise KeyError("Each element must define 'node_i' and 'node_j'.")
        ni = int(elem['node_i'])
        nj = int(elem['node_j'])
        if ni < 0 or ni >= n_nodes or nj < 0 or (nj >= n_nodes):
            raise IndexError('Element node index out of range.')
        (xi, yi, zi) = node_coords[ni]
        (xj, yj, zj) = node_coords[nj]
        try:
            E = float(elem['E'])
            nu = float(elem['nu'])
            A = float(elem['A'])
            J = float(elem['J'])
        except KeyError as e:
            raise KeyError(f'Element missing property {e}')
        if 'I_y' in elem:
            Iy = float(elem['I_y'])
        elif 'Iy' in elem:
            Iy = float(elem['Iy'])
        else:
            raise KeyError("Element missing 'I_y' (or 'Iy').")
        if 'I_z' in elem:
            Iz = float(elem['I_z'])
        elif 'Iz' in elem:
            Iz = float(elem['Iz'])
        else:
            raise KeyError("Element missing 'I_z' (or 'Iz').")
        dx = xj - xi
        dy = yj - yi
        dz = zj - zi
        L = np.sqrt(dx * dx + dy * dy + dz * dz)
        if np.isclose(L, 0.0):
            raise ValueError('Element has zero length.')
        k_local = local_elastic_stiffness_matrix_3D_beam(E, nu, A, L, Iy, Iz, J)
        ref_vec = elem.get('local_z', None)
        Gamma = beam_transformation_matrix_3D(xi, yi, zi, xj, yj, zj, ref_vec)
        k_global = Gamma.T @ k_local @ Gamma
        dofs = [ni * 6 + i for i in range(6)] + [nj * 6 + i for i in range(6)]
        for a in range(12):
            for b in range(12):
                K[dofs[a], dofs[b]] += k_global[a, b]
    return K