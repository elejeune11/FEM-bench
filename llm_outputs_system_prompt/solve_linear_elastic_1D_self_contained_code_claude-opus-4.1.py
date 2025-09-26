def solve_linear_elastic_1D_self_contained(x_min: float, x_max: float, num_elements: int, material_regions: List[Dict[str, float]], body_force_fn: Callable[[float], float], dirichlet_bc_list: List[Dict[str, float]], neumann_bc_list: Optional[List[Dict[str, float]]], n_gauss: int) -> Dict[str, np.ndarray]:
    """
    Solve a 1D linear elastic finite element problem with integrated meshing.
    Parameters:
        x_min (float): Start coordinate of the domain.
        x_max (float): End coordinate of the domain.
        num_elements (int): Number of linear elements.
        material_regions (List[Dict]): List of material regions, each with keys:
            "coord_min", "coord_max", "E", "A".
        body_force_fn (Callable): Function f(x) for body force.
        dirichlet_bc_list (List[Dict]): Each dict must contain:
            {
                "x_location": float,      # coordinate of prescribed node
                "u_prescribed": float     # displacement value
            }
        neumann_bc_list (Optional[List[Dict]]): Each dict must contain:
            {
                "x_location": float,  # coordinate of the node
                "load_mag": float     # magnitude of point load (positive = outward)
            }
        n_gauss (int): Number of Gauss points for numerical integration (1 to 3 supported).
    Returns:
        dict: Dictionary containing solution results:
    """
    n_nodes = num_elements + 1
    node_coords = np.linspace(x_min, x_max, n_nodes)
    K_global = np.zeros((n_nodes, n_nodes))
    F_global = np.zeros(n_nodes)
    gauss_points_xi = {1: np.array([0.0]), 2: np.array([-1.0 / np.sqrt(3), 1.0 / np.sqrt(3)]), 3: np.array([-np.sqrt(3.0 / 5.0), 0.0, np.sqrt(3.0 / 5.0)])}
    gauss_weights = {1: np.array([2.0]), 2: np.array([1.0, 1.0]), 3: np.array([5.0 / 9.0, 8.0 / 9.0, 5.0 / 9.0])}
    xi_points = gauss_points_xi[n_gauss]
    weights = gauss_weights[n_gauss]
    for elem_idx in range(num_elements):
        node1_idx = elem_idx
        node2_idx = elem_idx + 1
        x1 = node_coords[node1_idx]
        x2 = node_coords[node2_idx]
        h = x2 - x1
        x_mid = (x1 + x2) / 2.0
        E = None
        A = None
        for region in material_regions:
            if region['coord_min'] <= x_mid <= region['coord_max']:
                E = region['E']
                A = region['A']
                break
        if E is None or A is None:
            raise ValueError(f'Element at x={x_mid} not covered by material regions')
        k_elem = E * A / h * np.array([[1, -1], [-1, 1]])
        f_elem = np.zeros(2)
        for (i, xi) in enumerate(xi_points):
            x_gauss = x1 + (xi + 1) * h / 2.0
            N1 = (1 - xi) / 2.0
            N2 = (1 + xi) / 2.0
            f_val = body_force_fn(x_gauss)
            jacobian = h / 2.0
            f_elem[0] += weights[i] * N1 * f_val * jacobian
            f_elem[1] += weights[i] * N2 * f_val * jacobian
        K_global[node1_idx, node1_idx] += k_elem[0, 0]
        K_global[node1_idx, node2_idx] += k_elem[0, 1]
        K_global[node2_idx, node1_idx] += k_elem[1, 0]
        K_global[node2_idx, node2_idx] += k_elem[1, 1]
        F_global[node1_idx] += f_elem[0]
        F_global[node2_idx] += f_elem[1]
    if neumann_bc_list:
        for bc in neumann_bc_list:
            x_loc = bc['x_location']
            load_mag = bc['load_mag']
            node_idx = np.argmin(np.abs(node_coords - x_loc))
            F_global[node_idx] += load_mag
    dirichlet_nodes = []
    dirichlet_values = []
    for bc in dirichlet_bc_list:
        x_loc = bc['x_location']
        u_val = bc['u_prescribed']
        node_idx = np.argmin(np.abs(node_coords - x_loc))
        dirichlet_nodes.append(node_idx)
        dirichlet_values.append(u_val)
    dirichlet_nodes = np.array(dirichlet_nodes, dtype=int)
    dirichlet_values = np.array(dirichlet_values)
    free_nodes = np.setdiff1d(np.arange(n_nodes), dirichlet_nodes)
    K_ff = K_global[np.ix_(free_nodes, free_nodes)]
    K_fc = K_global[np.ix_(free_nodes, dirichlet_nodes)]
    F_f = F_global[free_nodes]
    F_f_modified = F_f - K_fc @ dirichlet_values
    u_free = np.linalg.solve(K_ff, F_f_modified)
    u_global = np.zeros(n_nodes)
    u_global[dirichlet_nodes] = dirichlet_values
    u_global[free_nodes] = u_free
    reactions = K_global[dirichlet_nodes, :] @ u_global - F_global[dirichlet_nodes]
    return {'displacements': u_global, 'reactions': reactions, 'node_coords': node_coords, 'reaction_nodes': dirichlet_nodes}