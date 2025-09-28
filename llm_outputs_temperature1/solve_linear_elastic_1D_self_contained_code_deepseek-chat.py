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
        node1 = elem
        node2 = elem + 1
        x1 = node_coords[node1]
        x2 = node_coords[node2]
        L = x2 - x1
        elem_center = (x1 + x2) / 2
        E = None
        A = None
        for region in material_regions:
            if region['coord_min'] <= elem_center <= region['coord_max']:
                E = region['E']
                A = region['A']
                break
        if E is None or A is None:
            raise ValueError(f'No material region found for element centered at {elem_center}')
        K_elem = np.zeros((2, 2))
        for i in range(n_gauss):
            xi = gauss_points[i]
            w = gauss_weights[i]
            N1 = 0.5 * (1 - xi)
            N2 = 0.5 * (1 + xi)
            dN1_dxi = -0.5
            dN2_dxi = 0.5
            J = L / 2
            dN1_dx = dN1_dxi / J
            dN2_dx = dN2_dxi / J
            B = np.array([dN1_dx, dN2_dx])
            K_elem += E * A * np.outer(B, B) * J * w
        K_global[node1:node2 + 1, node1:node2 + 1] += K_elem
        F_elem_body = np.zeros(2)
        for i in range(n_gauss):
            xi = gauss_points[i]
            w = gauss_weights[i]
            N1 = 0.5 * (1 - xi)
            N2 = 0.5 * (1 + xi)
            x = x1 + (xi + 1) * L / 2
            J = L / 2
            f_x = body_force_fn(x)
            F_elem_body[0] += f_x * N1 * J * w
            F_elem_body[1] += f_x * N2 * J * w
        F_global[node1] += F_elem_body[0]
        F_global[node2] += F_elem_body[1]
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
        u_prescribed = bc['u_prescribed']
        node_idx = np.argmin(np.abs(node_coords - x_loc))
        dirichlet_nodes.append(node_idx)
        prescribed_displacements.append(u_prescribed)
    penalty = 1000000000000.0 * np.max(np.abs(K_global))
    for (i, node) in enumerate(dirichlet_nodes):
        K_global[node, node] += penalty
        F_global[node] += penalty * prescribed_displacements[i]
    displacements = np.linalg.solve(K_global, F_global)
    reactions = np.zeros(len(dirichlet_nodes))
    for (i, node) in enumerate(dirichlet_nodes):
        reaction = np.dot(K_global[node, :], displacements) - F_global[node]
        reactions[i] = reaction
    return {'displacements': displacements, 'reactions': reactions, 'node_coords': node_coords, 'reaction_nodes': np.array(dirichlet_nodes)}