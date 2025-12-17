def MSA_3D_assemble_global_linear_elastic_stiffness_CC0_H2_T3(node_coords, elements):
    """
    Assemble the global stiffness matrix for a 3D linear-elastic frame structure composed of beam elements.
    Each 2-node Euler–Bernoulli beam contributes a 12×12 local elastic stiffness matrix
    based on its material and geometric properties (E, ν, A, I_y, I_z, J, L).
    The local matrix is expressed in the element’s coordinate system and includes
    axial, bending, and torsional stiffness terms.
    For global assembly, each local stiffness is transformed into global coordinates
    using a 12×12 direction-cosine transformation matrix (Γ) derived from the element’s
    start and end node coordinates and, if provided, an orientation vector (local_z).
    The global stiffness matrix K is formed by adding these transformed
    element matrices into the appropriate degree-of-freedom positions.
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
                Unit vector in global coordinates defining the local z-axis orientation.
                Must not be parallel to the beam axis. If None, a default is chosen.
                Default local_z = global z, unless the beam lies along global z — then default local_z = global y.
    Returns
    -------
    K : ndarray of shape (6 * n_nodes, 6 * n_nodes)
        The assembled global stiffness matrix of the structure, with 6 degrees of freedom per node.
    Notes
    -----
    """
    import numpy as np
    coords = np.asarray(node_coords, dtype=float)
    if coords.ndim != 2 or coords.shape[1] != 3:
        raise ValueError('node_coords must be an array of shape (n_nodes, 3)')
    n_nodes = coords.shape[0]
    ndof = 6 * n_nodes
    K = np.zeros((ndof, ndof), dtype=float)
    tol = 1e-12

    def make_rotation(ri, rj, local_z_ref):
        d = rj - ri
        L = np.linalg.norm(d)
        if not np.isfinite(L) or L <= tol:
            raise ValueError('Element length must be positive and finite.')
        ex = d / L
        if local_z_ref is None:
            zref = np.array([0.0, 0.0, 1.0], dtype=float)
            if abs(np.dot(ex, zref)) > 1.0 - 1e-08:
                zref = np.array([0.0, 1.0, 0.0], dtype=float)
        else:
            zref = np.asarray(local_z_ref, dtype=float).ravel()
            if zref.size != 3:
                raise ValueError('local_z must be array-like with 3 components.')
            nrm = np.linalg.norm(zref)
            if not np.isfinite(nrm) or nrm <= tol:
                raise ValueError('local_z must be a non-zero vector.')
            zref = zref / nrm
            if abs(np.dot(ex, zref)) > 1.0 - 1e-08:
                raise ValueError('local_z must not be parallel to the element axis.')
        ez_temp = zref - np.dot(zref, ex) * ex
        n_ez = np.linalg.norm(ez_temp)
        if n_ez <= tol:
            raise ValueError('Failed to construct a valid local z-axis.')
        ez = ez_temp / n_ez
        ey = np.cross(ez, ex)
        n_ey = np.linalg.norm(ey)
        if n_ey <= tol:
            raise ValueError('Failed to construct a valid local y-axis.')
        ey = ey / n_ey
        R = np.column_stack((ex, ey, ez))
        return (R, L)
    for elem in elements:
        i = int(elem['node_i'])
        j = int(elem['node_j'])
        if i < 0 or i >= n_nodes or j < 0 or (j >= n_nodes):
            raise IndexError('Element node indices out of range.')
        ri = coords[i]
        rj = coords[j]
        E = float(elem['E'])
        nu = float(elem['nu'])
        A = float(elem['A'])
        Iy = float(elem['I_y'])
        Iz = float(elem['I_z'])
        J = float(elem['J'])
        local_z = elem.get('local_z', None)
        R, L = make_rotation(ri, rj, local_z)
        G = E / (2.0 * (1.0 + nu))
        Kl = np.zeros((12, 12), dtype=float)
        k_ax = E * A / L
        Kl[0, 0] += k_ax
        Kl[0, 6] -= k_ax
        Kl[6, 0] -= k_ax
        Kl[6, 6] += k_ax
        k_tor = G * J / L
        Kl[3, 3] += k_tor
        Kl[3, 9] -= k_tor
        Kl[9, 3] -= k_tor
        Kl[9, 9] += k_tor
        EIz = E * Iz
        a = 12.0 * EIz / L ** 3
        b = 6.0 * EIz / L ** 2
        c = 4.0 * EIz / L
        d = 2.0 * EIz / L
        Kl[1, 1] += a
        Kl[1, 5] += b
        Kl[1, 7] -= a
        Kl[1, 11] += b
        Kl[5, 1] += b
        Kl[5, 5] += c
        Kl[5, 7] -= b
        Kl[5, 11] += d
        Kl[7, 1] -= a
        Kl[7, 5] -= b
        Kl[7, 7] += a
        Kl[7, 11] -= b
        Kl[11, 1] += b
        Kl[11, 5] += d
        Kl[11, 7] -= b
        Kl[11, 11] += c
        EIy = E * Iy
        a2 = 12.0 * EIy / L ** 3
        b2 = 6.0 * EIy / L ** 2
        c2 = 4.0 * EIy / L
        d2 = 2.0 * EIy / L
        Kl[2, 2] += a2
        Kl[2, 4] -= b2
        Kl[2, 8] -= a2
        Kl[2, 10] -= b2
        Kl[4, 2] -= b2
        Kl[4, 4] += c2
        Kl[4, 8] += b2
        Kl[4, 10] += d2
        Kl[8, 2] -= a2
        Kl[8, 4] += b2
        Kl[8, 8] += a2
        Kl[8, 10] += b2
        Kl[10, 2] -= b2
        Kl[10, 4] += d2
        Kl[10, 8] += b2
        Kl[10, 10] += c2
        Tn = np.zeros((6, 6), dtype=float)
        Tn[:3, :3] = R
        Tn[3:, 3:] = R
        Gamma = np.zeros((12, 12), dtype=float)
        Gamma[:6, :6] = Tn
        Gamma[6:, 6:] = Tn
        Kg = Gamma.T @ Kl @ Gamma
        dofs_i = np.arange(6 * i, 6 * i + 6)
        dofs_j = np.arange(6 * j, 6 * j + 6)
        dofs = np.concatenate((dofs_i, dofs_j))
        K[np.ix_(dofs, dofs)] += Kg
    return K