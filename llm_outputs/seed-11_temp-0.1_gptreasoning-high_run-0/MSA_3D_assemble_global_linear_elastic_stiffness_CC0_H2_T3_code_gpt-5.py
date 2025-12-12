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
    from typing import Callable, Optional
    import pytest
    node_coords = np.asarray(node_coords, dtype=float)
    n_nodes = int(node_coords.shape[0])
    ndof = 6 * n_nodes
    K = np.zeros((ndof, ndof), dtype=float)
    if len(elements) == 0:
        return K
    indices = []
    for e in elements:
        indices.append(int(e['node_i']))
        indices.append(int(e['node_j']))
    min_idx = min(indices)
    max_idx = max(indices)
    if min_idx >= 1 and max_idx <= n_nodes:
        base = 1
    else:
        base = 0
    tol = 1e-12
    global_y = np.array([0.0, 1.0, 0.0])
    global_z = np.array([0.0, 0.0, 1.0])
    for el in elements:
        i = int(el['node_i']) - base
        j = int(el['node_j']) - base
        xi = node_coords[i]
        xj = node_coords[j]
        dx = xj - xi
        L = float(np.linalg.norm(dx))
        if L <= tol:
            continue
        E = float(el['E'])
        nu = float(el['nu'])
        A = float(el['A'])
        Iy = float(el['I_y'])
        Iz = float(el['I_z'])
        J = float(el['J'])
        local_z = el.get('local_z', None)
        ex = dx / L
        if local_z is None:
            v = global_z.copy()
            if np.linalg.norm(np.cross(ex, v)) < 1e-08:
                v = global_y.copy()
        else:
            v = np.asarray(local_z, dtype=float)
            nv = np.linalg.norm(v)
            if nv > tol:
                v = v / nv
            else:
                v = global_z.copy()
        if np.linalg.norm(np.cross(ex, v)) < 1e-08:
            v = global_y.copy()
            if np.linalg.norm(np.cross(ex, v)) < 1e-08:
                v = global_z.copy()
        ez_temp = v - np.dot(v, ex) * ex
        n_ez = np.linalg.norm(ez_temp)
        if n_ez <= 1e-12:
            v2 = global_z if abs(ex[2]) < 0.9 else global_y
            ez_temp = v2 - np.dot(v2, ex) * ex
            n_ez = np.linalg.norm(ez_temp)
        if n_ez <= 1e-12:
            tmp = np.array([1.0, 0.0, 0.0]) if abs(ex[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
            ez_temp = tmp - np.dot(tmp, ex) * ex
            n_ez = np.linalg.norm(ez_temp)
        ez = ez_temp / n_ez
        ey = np.cross(ez, ex)
        ny = np.linalg.norm(ey)
        if ny <= 1e-12:
            ey = np.cross(ex, ez)
            ny = np.linalg.norm(ey)
        ey = ey / ny
        ez = np.cross(ex, ey)
        R = np.column_stack((ex, ey, ez))
        RT = R.T
        Gamma = np.zeros((12, 12), dtype=float)
        for blk in range(4):
            s = 3 * blk
            Gamma[s:s + 3, s:s + 3] = RT
        Kl = np.zeros((12, 12), dtype=float)
        EA_L = E * A / L
        G = E / (2.0 * (1.0 + nu))
        GJ_L = G * J / L
        L2 = L * L
        L3 = L2 * L
        Kl[0, 0] = EA_L
        Kl[0, 6] = -EA_L
        Kl[6, 0] = -EA_L
        Kl[6, 6] = EA_L
        Kl[3, 3] = GJ_L
        Kl[3, 9] = -GJ_L
        Kl[9, 3] = -GJ_L
        Kl[9, 9] = GJ_L
        kz = E * Iz / L3
        Kbz = kz * np.array([[12.0, 6.0 * L, -12.0, 6.0 * L], [6.0 * L, 4.0 * L2, -6.0 * L, 2.0 * L2], [-12.0, -6.0 * L, 12.0, -6.0 * L], [6.0 * L, 2.0 * L2, -6.0 * L, 4.0 * L2]])
        idx_z = [1, 5, 7, 11]
        for a in range(4):
            for b in range(4):
                Kl[idx_z[a], idx_z[b]] += Kbz[a, b]
        ky = E * Iy / L3
        Kby = ky * np.array([[12.0, -6.0 * L, -12.0, -6.0 * L], [-6.0 * L, 4.0 * L2, 6.0 * L, 2.0 * L2], [-12.0, 6.0 * L, 12.0, 6.0 * L], [-6.0 * L, 2.0 * L2, 6.0 * L, 4.0 * L2]])
        idx_y = [2, 4, 8, 10]
        for a in range(4):
            for b in range(4):
                Kl[idx_y[a], idx_y[b]] += Kby[a, b]
        Kg = Gamma.T @ Kl @ Gamma
        dofs_i = np.arange(6 * i, 6 * i + 6)
        dofs_j = np.arange(6 * j, 6 * j + 6)
        elem_dofs = np.concatenate((dofs_i, dofs_j))
        K[np.ix_(elem_dofs, elem_dofs)] += Kg
    return K