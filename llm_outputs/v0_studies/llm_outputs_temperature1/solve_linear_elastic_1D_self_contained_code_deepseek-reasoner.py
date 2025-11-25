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
    elements = []
    for i in range(num_elements):
        elements.append([i, i + 1])
    K_global = np.zeros((n_nodes, n_nodes))
    F_global = np.zeros(n_nodes)
    if n_gauss == 1:
        gauss_points = [0.0]
        gauss_weights = [2.0]
    elif n_gauss == 2:
        gauss_points = [-1 / np.sqrt(3), 1 / np.sqrt(3)]
        gauss_weights = [1.0, 1.0]
    elif n_gauss == 3:
        gauss_points = [-np.sqrt(3 / 5), 0, np.sqrt(3 / 5)]
        gauss_weights = [5 / 9, 8 / 9, 5 / 9]
    else:
        raise ValueError('n_gauss must be 1, 2, or 3')
    for (elem_idx, elem_nodes) in enumerate(elements):
        (node1, node2) = elem_nodes
        (x1, x2) = (node_coords[node1], node_coords[node2])
        L = x2 - x1
        element_center = (x1 + x2) / 2
        (E, A) = (None, None)
        for region in material_regions:
            if region['coord_min'] <= element_center <= region['coord_max']:
                E = region['E']
                A = region['A']
                break
        if E is None or A is None:
            raise ValueError(f'No material region found for element centered at {element_center}')
        K_elem = np.zeros((2, 2))
        for (gp, weight) in zip(gauss_points, gauss_weights):
            xi = gp
            N1 = 0.5 * (1 - xi)
            N2 = 0.5 * (1 + xi)
            dN1_dxi = -0.5
            dN2_dxi = 0.5
            J = L / 2
            invJ = 1 / J
            B = np.array([dN1_dxi * invJ, dN2_dxi * invJ])
            K_elem += E * A * np.outer(B, B) * J * weight
        K_global[node1:node2 + 1, node1:node2 + 1] += K_elem
        for (gp, weight) in zip(gauss_points, gauss_weights):
            xi = gp
            N1 = 0.5 * (1 - xi)
            N2 = 0.5 * (1 + xi)
            x_gp = N1 * x1 + N2 * x2
            J = L / 2
            b = body_force_fn(x_gp)
            F_elem = np.array([N1 * b, N2 * b]) * A * J * weight
            F_global[node1] += F_elem[0]
            F_global[node2] += F_elem[1]
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
    free_nodes = [i for i in range(n_nodes) if i not in dirichlet_nodes]
    K_ff = K_global[np.ix_(free_nodes, free_nodes)]
    K_fd = K_global[np.ix_(free_nodes, dirichlet_nodes)]
    F_f = F_global[free_nodes]
    U_d = np.array(prescribed_displacements)
    U_f = np.linalg.solve(K_ff, F_f - K_fd @ U_d)
    displacements = np.zeros(n_nodes)
    displacements[free_nodes] = U_f
    displacements[dirichlet_nodes] = U_d
    reactions = K_global[dirichlet_nodes] @ displacements - F_global[dirichlet_nodes]
    return {'displacements': displacements, 'reactions': reactions, 'node_coords': node_coords, 'reaction_nodes': np.array(dirichlet_nodes)}