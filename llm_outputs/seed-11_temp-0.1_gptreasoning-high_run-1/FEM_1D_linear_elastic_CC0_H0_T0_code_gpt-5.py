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
    if not x_max > x_min:
        raise ValueError('x_max must be greater than x_min.')
    if not isinstance(num_elements, int) or num_elements < 1:
        raise ValueError('num_elements must be a positive integer.')
    if n_gauss not in {1, 2, 3}:
        raise ValueError('n_gauss must be one of {1, 2, 3}.')
    n_nodes = num_elements + 1
    node_coords = np.linspace(x_min, x_max, n_nodes)
    L_total = x_max - x_min
    h = L_total / num_elements
    required_keys = {'coord_min', 'coord_max', 'E', 'A'}
    for r in material_regions:
        if not required_keys.issubset(r.keys()):
            raise ValueError("Each material region must include keys: 'coord_min', 'coord_max', 'E', 'A'.")
    E_e = np.zeros(num_elements, dtype=float)
    A_e = np.zeros(num_elements, dtype=float)
    atol_region = 1e-12 * max(1.0, abs(L_total))
    for e in range(num_elements):
        x1 = node_coords[e]
        x2 = node_coords[e + 1]
        x_center = 0.5 * (x1 + x2)
        matches = []
        for i, r in enumerate(material_regions):
            cmin = float(r['coord_min'])
            cmax = float(r['coord_max'])
            if x_center >= cmin - atol_region and x_center <= cmax + atol_region:
                matches.append(i)
        if len(matches) != 1:
            raise ValueError('Each element must be covered by exactly one material region.')
        r = material_regions[matches[0]]
        E_e[e] = float(r['E'])
        A_e[e] = float(r['A'])
    K = np.zeros((n_nodes, n_nodes), dtype=float)
    F = np.zeros(n_nodes, dtype=float)
    if n_gauss == 1:
        xi_gp = np.array([0.0], dtype=float)
        w_gp = np.array([2.0], dtype=float)
    elif n_gauss == 2:
        xi_gp = np.array([-np.sqrt(1.0 / 3.0), np.sqrt(1.0 / 3.0)], dtype=float)
        w_gp = np.array([1.0, 1.0], dtype=float)
    else:
        xi_gp = np.array([-np.sqrt(3.0 / 5.0), 0.0, np.sqrt(3.0 / 5.0)], dtype=float)
        w_gp = np.array([5.0 / 9.0, 8.0 / 9.0, 5.0 / 9.0], dtype=float)
    for e in range(num_elements):
        n1 = e
        n2 = e + 1
        x1 = node_coords[n1]
        x2 = node_coords[n2]
        Le = x2 - x1
        EA_over_L = E_e[e] * A_e[e] / Le
        ke = EA_over_L * np.array([[1.0, -1.0], [-1.0, 1.0]], dtype=float)
        K[n1, n1] += ke[0, 0]
        K[n1, n2] += ke[0, 1]
        K[n2, n1] += ke[1, 0]
        K[n2, n2] += ke[1, 1]
        fe = np.zeros(2, dtype=float)
        J = Le / 2.0
        for gp in range(xi_gp.size):
            xi = xi_gp[gp]
            w = w_gp[gp]
            N1 = 0.5 * (1.0 - xi)
            N2 = 0.5 * (1.0 + xi)
            x_g = N1 * x1 + N2 * x2
            f_g = float(body_force_fn(float(x_g)))
            fe[0] += N1 * f_g * J * w
            fe[1] += N2 * f_g * J * w
        F[n1] += fe[0]
        F[n2] += fe[1]
    if neumann_bc_list is not None:
        atol_node = max(1e-12 * max(1.0, abs(L_total)), 1e-12 * (abs(h) + 1.0))
        for bc in neumann_bc_list:
            if not isinstance(bc, dict) or ('x_location' not in bc or 'load_mag' not in bc):
                raise ValueError("Each Neumann BC must have 'x_location' and 'load_mag'.")
            xloc = float(bc['x_location'])
            load = float(bc['load_mag'])
            matches = np.where(np.isclose(node_coords, xloc, rtol=0.0, atol=atol_node))[0]
            if matches.size != 1:
                raise ValueError('Neumann BC x_location does not match a unique node.')
            node = int(matches[0])
            F[node] += load
    atol_node = max(1e-12 * max(1.0, abs(L_total)), 1e-12 * (abs(h) + 1.0))
    reaction_nodes_list: List[int] = []
    prescribed_u_map: Dict[int, float] = {}
    for bc in dirichlet_bc_list:
        if not isinstance(bc, dict) or ('x_location' not in bc or 'u_prescribed' not in bc):
            raise ValueError("Each Dirichlet BC must have 'x_location' and 'u_prescribed'.")
        xloc = float(bc['x_location'])
        up = float(bc['u_prescribed'])
        matches = np.where(np.isclose(node_coords, xloc, rtol=0.0, atol=atol_node))[0]
        if matches.size != 1:
            raise ValueError('Dirichlet BC x_location does not match a unique node.')
        node = int(matches[0])
        if node in prescribed_u_map:
            if not np.isclose(prescribed_u_map[node], up, rtol=0.0, atol=1e-12 * max(1.0, abs(up))):
                raise ValueError('Conflicting Dirichlet values at the same node.')
        else:
            prescribed_u_map[node] = up
            reaction_nodes_list.append(node)
    reaction_nodes = np.array(reaction_nodes_list, dtype=int)
    all_nodes = np.arange(n_nodes, dtype=int)
    if reaction_nodes.size > 0:
        free_mask = np.ones(n_nodes, dtype=bool)
        free_mask[reaction_nodes] = False
        free_nodes = all_nodes[free_mask]
    else:
        free_nodes = all_nodes
    u = np.zeros(n_nodes, dtype=float)
    for node, up in prescribed_u_map.items():
        u[node] = up
    if free_nodes.size > 0:
        K_ff = K[np.ix_(free_nodes, free_nodes)]
        if reaction_nodes.size > 0:
            K_fc = K[np.ix_(free_nodes, reaction_nodes)]
            rhs = F[free_nodes] - K_fc @ u[reaction_nodes]
        else:
            rhs = F[free_nodes]
        u_free = np.linalg.solve(K_ff, rhs)
        u[free_nodes] = u_free
    if reaction_nodes.size > 0:
        reactions_full = K @ u - F
        reactions = reactions_full[reaction_nodes]
    else:
        reactions = np.zeros(0, dtype=float)
    return {'displacements': u, 'reactions': reactions, 'node_coords': node_coords, 'reaction_nodes': reaction_nodes}