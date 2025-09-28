def assemble_global_stiffness_matrix_linear_elastic_3D(node_coords, elements):
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
    ndofs = 6 * n_nodes
    K = np.zeros((ndofs, ndofs), dtype=float)
    for (idx, el) in enumerate(elements):
        try:
            ni = int(el['node_i'])
            nj = int(el['node_j'])
        except Exception as e:
            raise ValueError(f"Element {idx} must contain 'node_i' and 'node_j'.") from e
        if not (0 <= ni < n_nodes and 0 <= nj < n_nodes):
            raise IndexError(f'Element {idx} references invalid node indices: {ni}, {nj}.')
        (xi, yi, zi) = coords[ni]
        (xj, yj, zj) = coords[nj]
        (dx, dy, dz) = (xj - xi, yj - yi, zj - zi)
        L = float(np.sqrt(dx * dx + dy * dy + dz * dz))
        if np.isclose(L, 0.0):
            raise ValueError(f'Element {idx} has zero length.')
        try:
            E = float(el['E'])
            nu = float(el['nu'])
            A = float(el['A'])
            Iy = float(el['I_y'])
            Iz = float(el['I_z'])
            J = float(el['J'])
        except KeyError as e:
            raise ValueError(f'Element {idx} missing required property: {e.args[0]}')
        except Exception as e:
            raise ValueError(f'Element {idx} has invalid property values.') from e
        local_z = el.get('local_z', None)
        if local_z is not None:
            local_z = np.asarray(local_z, dtype=float)
        k_local = local_elastic_stiffness_matrix_3D_beam(E, nu, A, L, Iy, Iz, J)
        Gamma = beam_transformation_matrix_3D(xi, yi, zi, xj, yj, zj, local_z)
        k_global_elem = Gamma.T @ k_local @ Gamma
        dofs_i = np.arange(6 * ni, 6 * ni + 6)
        dofs_j = np.arange(6 * nj, 6 * nj + 6)
        dofs = np.concatenate((dofs_i, dofs_j))
        K[np.ix_(dofs, dofs)] += k_global_elem
    return K