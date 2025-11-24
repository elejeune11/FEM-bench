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
    if n_gauss == 1:
        gauss_points = [0.0]
        gauss_weights = [2.0]
    elif n_gauss == 2:
        gauss_points = [-1 / np.sqrt(3), 1 / np.sqrt(3)]
        gauss_weights = [1.0, 1.0]
    elif n_gauss == 3:
        gauss_points = [-np.sqrt(3 / 5), 0.0, np.sqrt(3 / 5)]
        gauss_weights = [5 / 9, 8 / 9, 5 / 9]
    else:
        raise ValueError('n_gauss must be 1, 2, or 3')
    for elem in range(num_elements):
        (node1, node2) = (elem, elem + 1)
        (x1, x2) = (node_coords[node1], node_coords[node2])
        L_e = x2 - x1
        x_mid = (x1 + x2) / 2
        (E, A) = (None, None)
        for region in material_regions:
            if region['coord_min'] <= x_mid <= region['coord_max']:
                E = region['E']
                A = region['A']
                break
        if E is None or A is None:
            raise ValueError(f'No material properties found for element at x={x_mid}')
        k_e = E * A / L_e * np.array([[1, -1], [-1, 1]])
        f_e = np.zeros(2)
        for (gp, gw) in zip(gauss_points, gauss_weights):
            xi = gp
            x = x1 + (xi + 1) * L_e / 2
            N1 = (1 - xi) / 2
            N2 = (1 + xi) / 2
            f_val = body_force_fn(x)
            f_e[0] += N1 * f_val * L_e / 2 * gw
            f_e[1] += N2 * f_val * L_e / 2 * gw
        indices = [node1, node2]
        for i in range(2):
            F_global[indices[i]] += f_e[i]
            for j in range(2):
                K_global[indices[i], indices[j]] += k_e[i, j]
    if neumann_bc_list:
        for bc in neumann_bc_list:
            node_idx = np.argmin(np.abs(node_coords - bc['x_location']))
            F_global[node_idx] += bc['load_mag']
    dirichlet_nodes = []
    dirichlet_values = []
    for bc in dirichlet_bc_list:
        node_idx = np.argmin(np.abs(node_coords - bc['x_location']))
        dirichlet_nodes.append(node_idx)
        dirichlet_values.append(bc['u_prescribed'])
    dirichlet_nodes = np.array(dirichlet_nodes, dtype=int)
    dirichlet_values = np.array(dirichlet_values)
    all_nodes = np.arange(n_nodes)
    free_nodes = np.setdiff1d(all_nodes, dirichlet_nodes)
    K_ff = K_global[np.ix_(free_nodes, free_nodes)]
    K_fp = K_global[np.ix_(free_nodes, dirichlet_nodes)]
    K_pf = K_global[np.ix_(dirichlet_nodes, free_nodes)]
    K_pp = K_global[np.ix_(dirichlet_nodes, dirichlet_nodes)]
    F_f = F_global[free_nodes]
    if len(free_nodes) > 0:
        u_f = np.linalg.solve(K_ff, F_f - K_fp @ dirichlet_values)
    else:
        u_f = np.array([])
    u_global = np.zeros(n_nodes)
    u_global[free_nodes] = u_f
    u_global[dirichlet_nodes] = dirichlet_values
    if len(dirichlet_nodes) > 0:
        reactions = K_pf @ u_f + K_pp @ dirichlet_values - F_global[dirichlet_nodes]
    else:
        reactions = np.array([])
    return {'displacements': u_global, 'reactions': reactions, 'node_coords': node_coords, 'reaction_nodes': dirichlet_nodes}