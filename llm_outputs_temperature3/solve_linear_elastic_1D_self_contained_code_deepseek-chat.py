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
    gauss_data = {1: {'points': [0.0], 'weights': [2.0]}, 2: {'points': [-1 / np.sqrt(3), 1 / np.sqrt(3)], 'weights': [1.0, 1.0]}, 3: {'points': [-np.sqrt(3 / 5), 0.0, np.sqrt(3 / 5)], 'weights': [5 / 9, 8 / 9, 5 / 9]}}
    if n_gauss not in gauss_data:
        raise ValueError('n_gauss must be 1, 2, or 3')
    gauss_points = gauss_data[n_gauss]['points']
    gauss_weights = gauss_data[n_gauss]['weights']
    n_nodes = num_elements + 1
    node_coords = np.linspace(x_min, x_max, n_nodes)
    element_connectivity = np.array([[i, i + 1] for i in range(num_elements)])
    K_global = np.zeros((n_nodes, n_nodes))
    F_global = np.zeros(n_nodes)
    element_E = np.zeros(num_elements)
    element_A = np.zeros(num_elements)
    for elem_idx in range(num_elements):
        x1 = node_coords[elem_idx]
        x2 = node_coords[elem_idx + 1]
        elem_center = (x1 + x2) / 2
        for region in material_regions:
            if region['coord_min'] <= elem_center <= region['coord_max']:
                element_E[elem_idx] = region['E']
                element_A[elem_idx] = region['A']
                break
    for elem_idx in range(num_elements):
        node1 = element_connectivity[elem_idx, 0]
        node2 = element_connectivity[elem_idx, 1]
        x1 = node_coords[node1]
        x2 = node_coords[node2]
        L = x2 - x1
        E = element_E[elem_idx]
        A = element_A[elem_idx]
        k_elem = E * A / L * np.array([[1, -1], [-1, 1]])
        K_global[node1:node2 + 1, node1:node2 + 1] += k_elem
        for (gp, weight) in zip(gauss_points, gauss_weights):
            xi = gp
            N = np.array([(1 - xi) / 2, (1 + xi) / 2])
            x_gp = N[0] * x1 + N[1] * x2
            J = L / 2
            f_gp = body_force_fn(x_gp)
            f_elem = f_gp * A * weight * J * N
            F_global[node1] += f_elem[0]
            F_global[node2] += f_elem[1]
    if neumann_bc_list is not None:
        for bc in neumann_bc_list:
            x_loc = bc['x_location']
            load_mag = bc['load_mag']
            node_idx = np.argmin(np.abs(node_coords - x_loc))
            F_global[node_idx] += load_mag
    dirichlet_nodes = []
    prescribed_values = []
    for bc in dirichlet_bc_list:
        x_loc = bc['x_location']
        u_prescribed = bc['u_prescribed']
        node_idx = np.argmin(np.abs(node_coords - x_loc))
        dirichlet_nodes.append(node_idx)
        prescribed_values.append(u_prescribed)
    free_nodes = [i for i in range(n_nodes) if i not in dirichlet_nodes]
    K_ff = K_global[np.ix_(free_nodes, free_nodes)]
    K_fd = K_global[np.ix_(free_nodes, dirichlet_nodes)]
    F_f = F_global[free_nodes]
    displacements = np.zeros(n_nodes)
    displacements[dirichlet_nodes] = prescribed_values
    if len(free_nodes) > 0:
        F_modified = F_f - K_fd @ displacements[dirichlet_nodes]
        displacements[free_nodes] = np.linalg.solve(K_ff, F_modified)
    reactions = K_global[dirichlet_nodes, :] @ displacements - F_global[dirichlet_nodes]
    return {'displacements': displacements, 'reactions': reactions, 'node_coords': node_coords, 'reaction_nodes': np.array(dirichlet_nodes)}