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
    if elements is None:
        return K
    elems = list(elements)
    if len(elems) == 0 or n_nodes == 0:
        return K
    idxs = []
    for e in elems:
        if 'node_i' not in e or 'node_j' not in e:
            raise KeyError("Each element must contain 'node_i' and 'node_j'.")
        idxs.extend([int(e['node_i']), int(e['node_j'])])
    min_idx = min(idxs)
    max_idx = max(idxs)
    use_zero_based = min_idx >= 0 and max_idx <= n_nodes - 1
    use_one_based = min_idx >= 1 and max_idx <= n_nodes
    if use_zero_based and use_one_based:
        offset = 0
    elif use_zero_based:
        offset = 0
    elif use_one_based:
        offset = -1
    else:
        raise IndexError('Element node indices out of valid range for provided node_coords.')
    for e in elems:
        ni = int(e['node_i']) + offset
        nj = int(e['node_j']) + offset
        if ni < 0 or nj < 0 or ni >= n_nodes or (nj >= n_nodes):
            raise IndexError('Element node index out of range.')
        xi, yi, zi = coords[ni]
        xj, yj, zj = coords[nj]
        dx, dy, dz = (xj - xi, yj - yi, zj - zi)
        L = float(np.sqrt(dx * dx + dy * dy + dz * dz))
        try:
            E = float(e['E'])
            nu = float(e['nu'])
            A = float(e['A'])
            J = float(e['J'])
        except KeyError as ex:
            raise KeyError(f"Element missing required property '{ex.args[0]}'.")
        Iy_val = e['I_y'] if 'I_y' in e else e['Iy'] if 'Iy' in e else e['IY'] if 'IY' in e else None
        Iz_val = e['I_z'] if 'I_z' in e else e['Iz'] if 'Iz' in e else e['IZ'] if 'IZ' in e else None
        if Iy_val is None or Iz_val is None:
            if Iy_val is None and Iz_val is None:
                raise KeyError("Element missing required properties 'I_y' and 'I_z'.")
            if Iy_val is None:
                raise KeyError("Element missing required property 'I_y'.")
            raise KeyError("Element missing required property 'I_z'.")
        Iy = float(Iy_val)
        Iz = float(Iz_val)
        local_z = e.get('local_z', None)
        if local_z is None and 'ref_vec' in e:
            local_z = e['ref_vec']
        if local_z is None and 'reference_vector' in e:
            local_z = e['reference_vector']
        ref_vec = None if local_z is None else np.asarray(local_z, dtype=float)
        k_local = local_elastic_stiffness_matrix_3D_beam(E, nu, A, L, Iy, Iz, J)
        Gamma = beam_transformation_matrix_3D(xi, yi, zi, xj, yj, zj, ref_vec)
        k_global_e = Gamma.T @ k_local @ Gamma
        dofs = list(range(6 * ni, 6 * ni + 6)) + list(range(6 * nj, 6 * nj + 6))
        K[np.ix_(dofs, dofs)] += k_global_e
    return K