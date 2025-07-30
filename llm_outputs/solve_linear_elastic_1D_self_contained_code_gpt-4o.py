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
    node_coords = np.linspace(x_min, x_max, num_elements + 1)
    n_nodes = len(node_coords)
    K_global = np.zeros((n_nodes, n_nodes))
    F_global = np.zeros(n_nodes)
    if n_gauss == 1:
        gauss_points = np.array([0.0])
        gauss_weights = np.array([2.0])
    elif n_gauss == 2:
        gauss_points = np.array([-1 / np.sqrt(3), 1 / np.sqrt(3)])
        gauss_weights = np.array([1.0, 1.0])
    elif n_gauss == 3:
        gauss_points = np.array([-np.sqrt(3 / 5), 0.0, np.sqrt(3 / 5)])
        gauss_weights = np.array([5 / 9, 8 / 9, 5 / 9])
    else:
        raise ValueError('n_gauss must be 1, 2, or 3.')
    for elem in range(num_elements):
        x1 = node_coords[elem]
        x2 = node_coords[elem + 1]
        length = x2 - x1
        for region in material_regions:
            if region['coord_min'] <= x1 and region['coord_max'] >= x2:
                E = region['E']
                A = region['A']
                break
        else:
            raise ValueError('Element does not fit within any material region.')
        k_local = E * A / length * np.array([[1, -1], [-1, 1]])
        f_local = np.zeros(2)
        for (i, xi) in enumerate(gauss_points):
            N1 = 0.5 * (1 - xi)
            N2 = 0.5 * (1 + xi)
            x_gauss = N1 * x1 + N2 * x2
            f_local += gauss_weights[i] * body_force_fn(x_gauss) * np.array([N1, N2]) * length / 2
        K_global[elem:elem + 2, elem:elem + 2] += k_local
        F_global[elem:elem + 2] += f_local
    if neumann_bc_list:
        for bc in neumann_bc_list:
            x_loc = bc['x_location']
            load_mag = bc['load_mag']
            node_index = np.argmin(np.abs(node_coords - x_loc))
            F_global[node_index] += load_mag
    dirichlet_nodes = []
    dirichlet_values = []
    for bc in dirichlet_bc_list:
        x_loc = bc['x_location']
        u_prescribed = bc['u_prescribed']
        node_index = np.argmin(np.abs(node_coords - x_loc))
        dirichlet_nodes.append(node_index)
        dirichlet_values.append(u_prescribed)
    free_nodes = np.setdiff1d(np.arange(n_nodes), dirichlet_nodes)
    K_ff = K_global[np.ix_(free_nodes, free_nodes)]
    K_fc = K_global[np.ix_(free_nodes, dirichlet_nodes)]
    F_f = F_global[free_nodes]
    u_c = np.array(dirichlet_values)
    u_f = np.linalg.solve(K_ff, F_f - K_fc @ u_c)
    displacements = np.zeros(n_nodes)
    displacements[free_nodes] = u_f
    displacements[dirichlet_nodes] = u_c
    reactions = K_global[dirichlet_nodes, :] @ displacements - F_global[dirichlet_nodes]
    return {'displacements': displacements, 'reactions': reactions, 'node_coords': node_coords, 'reaction_nodes': np.array(dirichlet_nodes)}