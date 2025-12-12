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
    if x_max <= x_min:
        raise ValueError('x_max must be greater than x_min.')
    if num_elements < 1:
        raise ValueError('num_elements must be >= 1.')
    if n_gauss is None:
        n_gauss = 2
    if n_gauss not in (1, 2, 3):
        raise ValueError('n_gauss must be 1, 2, or 3.')
    n_nodes = num_elements + 1
    node_coords = np.linspace(x_min, x_max, n_nodes, dtype=float)
    L = x_max - x_min
    tol = 1e-12 * max(1.0, abs(L))
    if n_gauss == 1:
        gauss_xi = np.array([0.0])
        gauss_w = np.array([2.0])
    elif n_gauss == 2:
        r = 1.0 / np.sqrt(3.0)
        gauss_xi = np.array([-r, r])
        gauss_w = np.array([1.0, 1.0])
    else:
        r = np.sqrt(3.0 / 5.0)
        gauss_xi = np.array([-r, 0.0, r])
        gauss_w = np.array([5.0 / 9.0, 8.0 / 9.0, 5.0 / 9.0])
    K = np.zeros((n_nodes, n_nodes), dtype=float)
    F = np.zeros(n_nodes, dtype=float)
    for e in range(num_elements):
        n1 = e
        n2 = e + 1
        x1 = node_coords[n1]
        x2 = node_coords[n2]
        h = x2 - x1
        if h <= 0:
            raise ValueError('Non-positive element length encountered.')
        x_center = 0.5 * (x1 + x2)
        matched = []
        for reg in material_regions:
            cmin = float(reg['coord_min'])
            cmax = float(reg['coord_max'])
            if x_center >= cmin - tol and x_center <= cmax + tol:
                matched.append(reg)
        if len(matched) != 1:
            raise ValueError('Each element must fall into exactly one material region.')
        reg = matched[0]
        E = float(reg['E'])
        A = float(reg['A'])
        ke_fac = E * A / h
        ke = ke_fac * np.array([[1.0, -1.0], [-1.0, 1.0]], dtype=float)
        K[n1, n1] += ke[0, 0]
        K[n1, n2] += ke[0, 1]
        K[n2, n1] += ke[1, 0]
        K[n2, n2] += ke[1, 1]
        fe = np.zeros(2, dtype=float)
        J = h / 2.0
        for (xi, w) in zip(gauss_xi, gauss_w):
            N1 = 0.5 * (1.0 - xi)
            N2 = 0.5 * (1.0 + xi)
            x_gp = N1 * x1 + N2 * x2
            f_val = float(body_force_fn(x_gp))
            contrib = f_val * J * w
            fe[0] += N1 * contrib
            fe[1] += N2 * contrib
        F[n1] += fe[0]
        F[n2] += fe[1]
    if neumann_bc_list is not None:
        for bc in neumann_bc_list:
            x_loc = float(bc['x_location'])
            load_mag = float(bc['load_mag'])
            matches = np.where(np.abs(node_coords - x_loc) <= tol)[0]
            if matches.size != 1:
                raise ValueError('Neumann BC location must match exactly one node.')
            F[matches[0]] += load_mag
    dirichlet_nodes = []
    dirichlet_vals = []
    seen = set()
    for bc in dirichlet_bc_list:
        x_loc = float(bc['x_location'])
        u_val = float(bc['u_prescribed'])
        matches = np.where(np.abs(node_coords - x_loc) <= tol)[0]
        if matches.size != 1:
            raise ValueError('Dirichlet BC location must match exactly one node.')
        idx = int(matches[0])
        if idx in seen:
            raise ValueError('Duplicate Dirichlet BC specified at the same node.')
        seen.add(idx)
        dirichlet_nodes.append(idx)
        dirichlet_vals.append(u_val)
    c = np.array(dirichlet_nodes, dtype=int)
    u_c = np.array(dirichlet_vals, dtype=float)
    u = np.zeros(n_nodes, dtype=float)
    if c.size > 0:
        u[c] = u_c
    all_idx = np.arange(n_nodes, dtype=int)
    if c.size > 0:
        free = np.setdiff1d(all_idx, c, assume_unique=False)
    else:
        free = all_idx.copy()
    if free.size > 0:
        K_rr = K[np.ix_(free, free)]
        rhs = F[free]
        if c.size > 0:
            K_rc = K[np.ix_(free, c)]
            rhs = rhs - K_rc @ u_c
        if free.size > 0:
            u_r = np.linalg.solve(K_rr, rhs)
            u[free] = u_r
    reactions = np.array([], dtype=float)
    if c.size > 0:
        reactions = (K @ u - F)[c]
    return {'displacements': u, 'reactions': reactions, 'node_coords': node_coords, 'reaction_nodes': c}