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
    if n_gauss not in {1, 2, 3}:
        raise ValueError(f'n_gauss must be in {{1, 2, 3}}, got {n_gauss}')
    gauss_data = {1: (np.array([0.0]), np.array([2.0])), 2: (np.array([-1.0 / np.sqrt(3.0), 1.0 / np.sqrt(3.0)]), np.array([1.0, 1.0])), 3: (np.array([-np.sqrt(3.0 / 5.0), 0.0, np.sqrt(3.0 / 5.0)]), np.array([5.0 / 9.0, 8.0 / 9.0, 5.0 / 9.0]))}
    (gauss_pts, gauss_wts) = gauss_data[n_gauss]
    num_nodes = num_elements + 1
    node_coords = np.linspace(x_min, x_max, num_nodes)
    h = (x_max - x_min) / num_elements
    K_global = np.zeros((num_nodes, num_nodes))
    F_global = np.zeros(num_nodes)
    element_E = np.zeros(num_elements)
    element_A = np.zeros(num_elements)
    for e in range(num_elements):
        x1 = node_coords[e]
        x2 = node_coords[e + 1]
        x_center = 0.5 * (x1 + x2)
        matching_regions = []
        for region in material_regions:
            if region['coord_min'] <= x_center <= region['coord_max']:
                matching_regions.append(region)
        if len(matching_regions) != 1:
            raise ValueError(f'Element {e} with center {x_center} is not covered by exactly one material region')
        element_E[e] = matching_regions[0]['E']
        element_A[e] = matching_regions[0]['A']
    for e in range(num_elements):
        x1 = node_coords[e]
        x2 = node_coords[e + 1]
        Le = x2 - x1
        E = element_E[e]
        A = element_A[e]
        ke = E * A / Le * np.array([[1.0, -1.0], [-1.0, 1.0]])
        K_global[e, e] += ke[0, 0]
        K_global[e, e + 1] += ke[0, 1]
        K_global[e + 1, e] += ke[1, 0]
        K_global[e + 1, e + 1] += ke[1, 1]
        fe = np.zeros(2)
        jacobian = Le / 2.0
        for gp in range(n_gauss):
            xi = gauss_pts[gp]
            w = gauss_wts[gp]
            N1 = 0.5 * (1.0 - xi)
            N2 = 0.5 * (1.0 + xi)
            x_phys = N1 * x1 + N2 * x2
            f_val = body_force_fn(x_phys)
            fe[0] += N1 * f_val * w * jacobian
            fe[1] += N2 * f_val * w * jacobian
        F_global[e] += fe[0]
        F_global[e + 1] += fe[1]
    if neumann_bc_list is not None:
        for nbc in neumann_bc_list:
            x_loc = nbc['x_location']
            load_mag = nbc['load_mag']
            tol = 1e-12 * (x_max - x_min)
            matching_nodes = np.where(np.abs(node_coords - x_loc) < tol)[0]
            if len(matching_nodes) != 1:
                raise ValueError(f'Neumann BC at x={x_loc} does not match a unique node')
            node_idx = matching_nodes[0]
            F_global[node_idx] += load_mag
    dirichlet_nodes = []
    dirichlet_values = []
    for dbc in dirichlet_bc_list:
        x_loc = dbc['x_location']
        u_val = dbc['u_prescribed']
        tol = 1e-12 * (x_max - x_min)
        matching_nodes = np.where(np.abs(node_coords - x_loc) < tol)[0]
        if len(matching_nodes) != 1:
            raise ValueError(f'Dirichlet BC at x={x_loc} does not match a unique node')
        node_idx = matching_nodes[0]
        dirichlet_nodes.append(node_idx)
        dirichlet_values.append(u_val)
    dirichlet_nodes = np.array(dirichlet_nodes, dtype=int)
    dirichlet_values = np.array(dirichlet_values)
    all_dofs = np.arange(num_nodes)
    free_dofs = np.setdiff1d(all_dofs, dirichlet_nodes)
    F_modified = F_global.copy()
    for (i, node) in enumerate(dirichlet_nodes):
        F_modified -= K_global[:, node] * dirichlet_values[i]
    K_reduced = K_global[np.ix_(free_dofs, free_dofs)]
    F_reduced = F_modified[free_dofs]
    u_free = np.linalg.solve(K_reduced, F_reduced)
    u_global = np.zeros(num_nodes)
    for (i, node) in enumerate(dirichlet_nodes):
        u_global[node] = dirichlet_values[i]
    u_global[free_dofs] = u_free
    reactions = np.zeros(len(dirichlet_nodes))
    for (i, node) in enumerate(dirichlet_nodes):
        reactions[i] = np.dot(K_global[node, :], u_global) - F_global[node]
    return {'displacements': u_global, 'reactions': reactions, 'node_coords': node_coords, 'reaction_nodes': dirichlet_nodes}