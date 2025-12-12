def FEM_1D_linear_elastic_CC0_H0_T0(x_min: float, x_max: float, num_elements: int, material_regions: List[Dict[str, float]], body_force_fn: Callable[[float], float], dirichlet_bc_list: List[Dict[str, float]], neumann_bc_list: Optional[List[Dict[str, float]]], n_gauss: int) -> Dict[str, np.ndarray]:
    """
    Solve a 1D small-strain, linear-elastic bar problem with built-in meshing and assembly.
    Builds a uniform 2-node (linear) mesh on [x_min, x_max], assigns element-wise
    material properties from piecewise regions, assembles the global stiffness and
    force vectors (including body forces via Gauss quadrature and optional point
    Neumann loads), applies Dirichlet constraints, solves for nodal displacements,
    and returns displacements and reactions.
    Parameters
    ----------
    x_min, x_max : float
        Domain bounds (x_max > x_min).
    num_elements : int
        Number of linear elements (num_nodes = num_elements + 1).
    material_regions : list of dict
        Piecewise-constant material regions. Each dict must contain:
        {"coord_min": float, "coord_max": float, "E": float, "A": float}.
        Elements are assigned by their midpoint x_center. Every element must fall
        into exactly one region.
    body_force_fn : Callable[[float], float]
        Body force density f(x) (force per unit length). Evaluated at quadrature points.
    dirichlet_bc_list : list of dict
        Dirichlet boundary conditions applied at existing mesh nodes. Each dict:
        {"x_location": float, "u_prescribed": float}.
    neumann_bc_list : list of dict, optional
        Point loads applied at existing mesh nodes. Each dict:
        {"x_location": float, "load_mag": float}.
        Positive load acts in the +x direction (outward).
    n_gauss : int, optional
        Number of Gauss points per element (1–3 supported). Default 2 (exact for
        linear elements with constant EA and linear f mapped through x(ξ)).
    Returns
    -------
    dict
        {
            "displacements": np.ndarray,  shape (n_nodes,)
                Nodal displacement vector.
            "reactions": np.ndarray,      shape (n_dirichlet,)
                Reaction forces at the Dirichlet-constrained nodes (in the order
                listed by `dirichlet_bc_list` / discovered nodes).
            "node_coords": np.ndarray,    shape (n_nodes,)
                Coordinates of all mesh nodes.
            "reaction_nodes": np.ndarray, shape (n_dirichlet,)
                Indices of nodes where reactions are reported.
        }
    Raises
    ------
    ValueError
        If an element is not covered by exactly one material region, if BC
        coordinates do not match a unique node, or if `n_gauss` is not in {1,2,3}.
    numpy.linalg.LinAlgError
        If the reduced stiffness matrix is singular (e.g., insufficient Dirichlet constraints).
    """
    import numpy as _np
    if x_max <= x_min:
        raise ValueError('x_max must be greater than x_min.')
    if num_elements < 1:
        raise ValueError('num_elements must be >= 1.')
    if n_gauss not in (1, 2, 3):
        raise ValueError('n_gauss must be 1, 2, or 3.')
    n_nodes = num_elements + 1
    node_coords = _np.linspace(x_min, x_max, n_nodes)
    element_nodes = [(i, i + 1) for i in range(num_elements)]
    elem_lengths = _np.diff(node_coords)
    elem_E = _np.empty(num_elements, dtype=float)
    elem_A = _np.empty(num_elements, dtype=float)
    tol = 1e-12 * (abs(x_max - x_min) + 1.0)
    for (e, (n1, n2)) in enumerate(element_nodes):
        x1 = node_coords[n1]
        x2 = node_coords[n2]
        x_center = 0.5 * (x1 + x2)
        matches = []
        for region in material_regions:
            coord_min = region['coord_min']
            coord_max = region['coord_max']
            if x_center >= coord_min - tol and x_center <= coord_max + tol:
                matches.append(region)
        if len(matches) != 1:
            raise ValueError(f'Element {e} at x={x_center} not covered by exactly one material region.')
        region = matches[0]
        elem_E[e] = float(region['E'])
        elem_A[e] = float(region['A'])
    K = _np.zeros((n_nodes, n_nodes), dtype=float)
    F = _np.zeros(n_nodes, dtype=float)
    if n_gauss == 1:
        gauss_xi = _np.array([0.0])
        gauss_w = _np.array([2.0])
    elif n_gauss == 2:
        a = 1.0 / _np.sqrt(3.0)
        gauss_xi = _np.array([-a, a])
        gauss_w = _np.array([1.0, 1.0])
    else:
        a = _np.sqrt(3.0 / 5.0)
        gauss_xi = _np.array([-a, 0.0, a])
        gauss_w = _np.array([5.0 / 9.0, 8.0 / 9.0, 5.0 / 9.0])

    def N_vals(xi):
        N1 = 0.5 * (1.0 - xi)
        N2 = 0.5 * (1.0 + xi)
        return _np.array([N1, N2])
    for (e, (n1, n2)) in enumerate(element_nodes):
        x1 = node_coords[n1]
        x2 = node_coords[n2]
        L = x2 - x1
        E = elem_E[e]
        A = elem_A[e]
        ke = E * A / L * _np.array([[1.0, -1.0], [-1.0, 1.0]], dtype=float)
        fe = _np.zeros(2, dtype=float)
        J = L / 2.0
        for (xi, w) in zip(gauss_xi, gauss_w):
            N = N_vals(xi)
            x_gp = N[0] * x1 + N[1] * x2
            fval = float(body_force_fn(x_gp))
            fe += w * N * fval * J
        K[n1:n2 + 1, n1:n2 + 1] += ke
        F[n1:n2 + 1] += fe
    if neumann_bc_list is not None:
        for nb in neumann_bc_list:
            x_loc = float(nb['x_location'])
            load = float(nb['load_mag'])
            matches = _np.where(_np.isclose(node_coords, x_loc, rtol=1e-08, atol=1e-12))[0]
            if matches.size != 1:
                raise ValueError(f'Neumann BC at x={x_loc} does not match a unique node.')
            idx = int(matches[0])
            F[idx] += load
    dirichlet_nodes = []
    dirichlet_values = []
    for db in dirichlet_bc_list:
        x_loc = float(db['x_location'])
        u_presc = float(db['u_prescribed'])
        matches = _np.where(_np.isclose(node_coords, x_loc, rtol=1e-08, atol=1e-12))[0]
        if matches.size != 1:
            raise ValueError(f'Dirichlet BC at x={x_loc} does not match a unique node.')
        idx = int(matches[0])
        if idx in dirichlet_nodes:
            raise ValueError(f'Multiple Dirichlet BCs specified for node at x={x_loc}.')
        dirichlet_nodes.append(idx)
        dirichlet_values.append(u_presc)
    dirichlet_nodes = _np.array(dirichlet_nodes, dtype=int)
    dirichlet_values = _np.array(dirichlet_values, dtype=float)
    all_nodes = _np.arange(n_nodes, dtype=int)
    if dirichlet_nodes.size > 0:
        mask = _np.ones(n_nodes, dtype=bool)
        mask[dirichlet_nodes] = False
        free_nodes = all_nodes[mask]
    else:
        free_nodes = all_nodes.copy()
    K_ff = K[_np.ix_(free_nodes, free_nodes)]
    K_fc = K[_np.ix_(free_nodes, dirichlet_nodes)] if dirichlet_nodes.size > 0 else _np.zeros((K_ff.shape[0], 0))
    K_cf = K[_np.ix_(dirichlet_nodes, free_nodes)] if dirichlet_nodes.size > 0 else _np.zeros((0, K_ff.shape[0]))
    K_cc = K[_np.ix_(dirichlet_nodes, dirichlet_nodes)] if dirichlet_nodes.size > 0 else _np.zeros((0, 0))
    F_f = F[free_nodes]
    F_c = F[dirichlet_nodes] if dirichlet_nodes.size > 0 else _np.zeros(0)
    rhs = F_f.copy()
    if dirichlet_nodes.size > 0:
        rhs = rhs - K_fc.dot(dirichlet_values)
    try:
        if free_nodes.size > 0:
            u_f = _np.linalg.solve(K_ff, rhs)
        else:
            u_f = _np.zeros(0, dtype=float)
    except _np.linalg.LinAlgError:
        raise
    displacements = _np.zeros(n_nodes, dtype=float)
    displacements[free_nodes] = u_f
    if dirichlet_nodes.size > 0:
        displacements[dirichlet_nodes] = dirichlet_values
    if dirichlet_nodes.size > 0:
        R_c = K_cf.dot(u_f) + K_cc.dot(dirichlet_values) - F_c
    else:
        R_c = _np.zeros(0, dtype=float)
    result = {'displacements': displacements, 'reactions': _np.array(R_c, dtype=float), 'node_coords': node_coords, 'reaction_nodes': _np.array(dirichlet_nodes, dtype=int)}
    return result