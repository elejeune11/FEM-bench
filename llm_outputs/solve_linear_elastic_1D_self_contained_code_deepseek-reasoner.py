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
    element_length = (x_max - x_min) / num_elements
    if n_gauss == 1:
        gauss_points = [0.0]
        gauss_weights = [2.0]
    elif n_gauss == 2:
        gauss_points = [-1.0 / np.sqrt(3.0), 1.0 / np.sqrt(3.0)]
        gauss_weights = [1.0, 1.0]
    elif n_gauss == 3:
        gauss_points = [-np.sqrt(3.0 / 5.0), 0.0, np.sqrt(3.0 / 5.0)]
        gauss_weights = [5.0 / 9.0, 8.0 / 9.0, 5.0 / 9.0]
    else:
        raise ValueError('n_gauss must be 1, 2, or 3')
    K_global = np.zeros((n_nodes, n_nodes))
    F_global = np.zeros(n_nodes)
    element_properties = []
    for i in range(num_elements):
        x_center = (node_coords[i] + node_coords[i + 1]) / 2.0
        (E, A) = (None, None)
        for region in material_regions:
            if region['coord_min'] <= x_center <= region['coord_max']:
                E = region['E']
                A = region['A']
                break
        if E is None or A is None:
            raise ValueError(f'Element {i} center {x_center} not covered by any material region')
        element_properties.append((E, A))
    for i in range(num_elements):
        (E, A) = element_properties[i]
        le = element_length
        (x1, x2) = (node_coords[i], node_coords[i + 1])
        k_e = E * A / le * np.array([[1, -1], [-1, 1]])
        K_global[i:i + 2, i:i + 2] += k_e
        f_e_body = np.zeros(2)
        for (gp, weight) in zip(gauss_points, gauss_weights):
            xi = gp
            N1 = (1 - xi) / 2
            N2 = (1 + xi) / 2
            x_gp = N1 * x1 + N2 * x2
            jacobian = le / 2
            b = body_force_fn(x_gp)
            f_e_body[0] += weight * N1 * b * jacobian
            f_e_body[1] += weight * N2 * b * jacobian
        F_global[i] += f_e_body[0]
        F_global[i + 1] += f_e_body[1]
    if neumann_bc_list is not None:
        for bc in neumann_bc_list:
            x_loc = bc['x_location']
            load = bc['load_mag']
            node_idx = np.argmin(np.abs(node_coords - x_loc))
            F_global[node_idx] += load
    dirichlet_nodes = []
    prescribed_displacements = []
    for bc in dirichlet_bc_list:
        x_loc = bc['x_location']
        u_pres = bc['u_prescribed']
        node_idx = np.argmin(np.abs(node_coords - x_loc))
        dirichlet_nodes.append(node_idx)
        prescribed_displacements.append(u_pres)
    dirichlet_nodes = np.array(dirichlet_nodes)
    prescribed_displacements = np.array(prescribed_displacements)
    penalty = 1000000000000.0 * np.max(np.abs(K_global))
    for (i, node_idx) in enumerate(dirichlet_nodes):
        K_global[node_idx, node_idx] += penalty
        F_global[node_idx] += penalty * prescribed_displacements[i]
    displacements = np.linalg.solve(K_global, F_global)
    reactions = np.zeros(len(dirichlet_nodes))
    F_int = K_global @ displacements
    for (i, node_idx) in enumerate(dirichlet_nodes):
        reactions[i] = F_int[node_idx] - F_global[node_idx]
    return {'displacements': displacements, 'reactions': reactions, 'node_coords': node_coords, 'reaction_nodes': dirichlet_nodes}