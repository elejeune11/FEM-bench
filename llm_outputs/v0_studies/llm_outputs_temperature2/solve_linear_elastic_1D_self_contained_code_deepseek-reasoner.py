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
    element_E = np.zeros(num_elements)
    element_A = np.zeros(num_elements)
    for i in range(num_elements):
        x_start = node_coords[i]
        x_end = node_coords[i + 1]
        x_center = (x_start + x_end) / 2
        for region in material_regions:
            if region['coord_min'] <= x_center <= region['coord_max']:
                element_E[i] = region['E']
                element_A[i] = region['A']
                break
    for elem in range(num_elements):
        node1 = elem
        node2 = elem + 1
        x1 = node_coords[node1]
        x2 = node_coords[node2]
        L = x2 - x1
        E = element_E[elem]
        A = element_A[elem]
        k_elem = E * A / L * np.array([[1, -1], [-1, 1]])
        K_global[node1, node1] += k_elem[0, 0]
        K_global[node1, node2] += k_elem[0, 1]
        K_global[node2, node1] += k_elem[1, 0]
        K_global[node2, node2] += k_elem[1, 1]
        f_elem = np.zeros(2)
        for (gp, weight) in zip(gauss_points, gauss_weights):
            xi = gp
            N1 = 0.5 * (1 - xi)
            N2 = 0.5 * (1 + xi)
            x = N1 * x1 + N2 * x2
            N = np.array([N1, N2])
            J = L / 2
            b = body_force_fn(x)
            f_elem += N * b * weight * J
        F_global[node1] += f_elem[0]
        F_global[node2] += f_elem[1]
    if neumann_bc_list is not None:
        for bc in neumann_bc_list:
            x_loc = bc['x_location']
            load = bc['load_mag']
            node_idx = np.argmin(np.abs(node_coords - x_loc))
            F_global[node_idx] += load
    dirichlet_nodes = []
    prescribed_values = []
    for bc in dirichlet_bc_list:
        x_loc = bc['x_location']
        u_prescribed = bc['u_prescribed']
        node_idx = np.argmin(np.abs(node_coords - x_loc))
        dirichlet_nodes.append(node_idx)
        prescribed_values.append(u_prescribed)
    dirichlet_nodes_sorted = np.sort(dirichlet_nodes)
    prescribed_values_sorted = [prescribed_values[np.where(dirichlet_nodes == node)[0][0]] for node in dirichlet_nodes_sorted]
    free_nodes = [i for i in range(n_nodes) if i not in dirichlet_nodes_sorted]
    K_ff = K_global[np.ix_(free_nodes, free_nodes)]
    K_fd = K_global[np.ix_(free_nodes, dirichlet_nodes_sorted)]
    F_f = F_global[free_nodes]
    U_d = np.array(prescribed_values_sorted)
    U_f = np.linalg.solve(K_ff, F_f - K_fd @ U_d)
    displacements = np.zeros(n_nodes)
    displacements[free_nodes] = U_f
    displacements[dirichlet_nodes_sorted] = U_d
    reactions = K_global[dirichlet_nodes_sorted] @ displacements - F_global[dirichlet_nodes_sorted]
    return {'displacements': displacements, 'reactions': reactions, 'node_coords': node_coords, 'reaction_nodes': np.array(dirichlet_nodes_sorted)}