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
    node_coords = np.asarray(node_coords, dtype=float)
    if node_coords.ndim != 2 or node_coords.shape[1] != 3:
        raise ValueError('node_coords must be an array of shape (n_nodes, 3).')
    n_nodes = node_coords.shape[0]
    ndof = 6 * n_nodes
    K = np.zeros((ndof, ndof), dtype=float)
    if elements is None or len(elements) == 0:
        return K
    idx_vals = []
    for el in elements:
        if 'node_i' not in el or 'node_j' not in el:
            raise KeyError("Each element must contain 'node_i' and 'node_j'.")
        idx_vals.extend([int(el['node_i']), int(el['node_j'])])
    min_idx = int(np.min(idx_vals))
    max_idx = int(np.max(idx_vals))
    if 0 <= min_idx and max_idx < n_nodes:
        shift = 0
    elif 1 <= min_idx and max_idx <= n_nodes:
        shift = -1
    else:
        raise IndexError('Element node indices are out of bounds for provided node_coords.')
    tol_len = 1e-12
    tol_parallel = 1e-10
    for el in elements:
        i = int(el['node_i']) + shift
        j = int(el['node_j']) + shift
        if not (0 <= i < n_nodes and 0 <= j < n_nodes):
            raise IndexError('Element node indices out of valid range after base adjustment.')
        xi = node_coords[i]
        xj = node_coords[j]
        dx = xj - xi
        L = float(np.linalg.norm(dx))
        if not np.isfinite(L) or L <= tol_len:
            raise ValueError('Element length must be positive and finite.')
        xhat = dx / L
        if 'local_z' in el and el['local_z'] is not None:
            zref = np.asarray(el['local_z'], dtype=float).reshape(-1)
            if zref.size != 3:
                raise ValueError('local_z must have exactly 3 components.')
            nz = np.linalg.norm(zref)
            if nz <= tol_len:
                raise ValueError('local_z must be non-zero.')
            zref = zref / nz
            if np.linalg.norm(np.cross(zref, xhat)) <= tol_parallel:
                raise ValueError('Provided local_z must not be parallel to beam axis.')
        else:
            z_global = np.array([0.0, 0.0, 1.0], dtype=float)
            y_global = np.array([0.0, 1.0, 0.0], dtype=float)
            if abs(float(np.dot(xhat, z_global))) >= 1.0 - tol_parallel:
                zref = y_global
            else:
                zref = z_global
        yhat = np.cross(zref, xhat)
        ny = np.linalg.norm(yhat)
        if ny <= tol_parallel:
            raise ValueError('local_z must not be parallel to the element axis.')
        yhat = yhat / ny
        zhat = np.cross(xhat, yhat)
        zhat = zhat / np.linalg.norm(zhat)
        R = np.vstack((xhat, yhat, zhat))
        T = np.zeros((12, 12), dtype=float)
        T[0:3, 0:3] = R
        T[3:6, 3:6] = R
        T[6:9, 6:9] = R
        T[9:12, 9:12] = R
        E = float(el['E'])
        nu = float(el['nu'])
        A = float(el['A'])
        Iy = float(el['I_y'])
        Iz = float(el['I_z'])
        J = float(el['J'])
        G = E / (2.0 * (1.0 + nu))
        Kl = np.zeros((12, 12), dtype=float)
        k_ax = E * A / L
        Kl[0, 0] += k_ax
        Kl[0, 6] += -k_ax
        Kl[6, 0] += -k_ax
        Kl[6, 6] += k_ax
        k_tor = G * J / L
        Kl[3, 3] += k_tor
        Kl[3, 9] += -k_tor
        Kl[9, 3] += -k_tor
        Kl[9, 9] += k_tor
        EIz = E * Iz
        c1 = 12.0 * EIz / L ** 3
        c2 = 6.0 * EIz / L ** 2
        c3 = 4.0 * EIz / L
        c4 = 2.0 * EIz / L
        iy_i, izr_i, iy_j, izr_j = (1, 5, 7, 11)
        Kl[iy_i, iy_i] += c1
        Kl[iy_i, izr_i] += c2
        Kl[iy_i, iy_j] += -c1
        Kl[iy_i, izr_j] += c2
        Kl[izr_i, iy_i] += c2
        Kl[izr_i, izr_i] += c3
        Kl[izr_i, iy_j] += -c2
        Kl[izr_i, izr_j] += c4
        Kl[iy_j, iy_i] += -c1
        Kl[iy_j, izr_i] += -c2
        Kl[iy_j, iy_j] += c1
        Kl[iy_j, izr_j] += -c2
        Kl[izr_j, iy_i] += c2
        Kl[izr_j, izr_i] += c4
        Kl[izr_j, iy_j] += -c2
        Kl[izr_j, izr_j] += c3
        EIy = E * Iy
        d1 = 12.0 * EIy / L ** 3
        d2 = 6.0 * EIy / L ** 2
        d3 = 4.0 * EIy / L
        d4 = 2.0 * EIy / L
        iz_i, iyr_i, iz_j, iyr_j = (2, 4, 8, 10)
        Kl[iz_i, iz_i] += d1
        Kl[iz_i, iyr_i] += -d2
        Kl[iz_i, iz_j] += -d1
        Kl[iz_i, iyr_j] += -d2
        Kl[iyr_i, iz_i] += -d2
        Kl[iyr_i, iyr_i] += d3
        Kl[iyr_i, iz_j] += d2
        Kl[iyr_i, iyr_j] += d4
        Kl[iz_j, iz_i] += -d1
        Kl[iz_j, iyr_i] += d2
        Kl[iz_j, iz_j] += d1
        Kl[iz_j, iyr_j] += d2
        Kl[iyr_j, iz_i] += -d2
        Kl[iyr_j, iyr_i] += d4
        Kl[iyr_j, iz_j] += d2
        Kl[iyr_j, iyr_j] += d3
        Ke = T.T @ Kl @ T
        dofs_i = np.arange(6 * i, 6 * i + 6)
        dofs_j = np.arange(6 * j, 6 * j + 6)
        dof_map = np.concatenate((dofs_i, dofs_j))
        K[np.ix_(dof_map, dof_map)] += Ke
    return K