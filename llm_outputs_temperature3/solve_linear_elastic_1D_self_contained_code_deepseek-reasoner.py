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
        xi_points = [0.0]
        weights = [2.0]
    elif n_gauss == 2:
        xi_points = [-1.0 / np.sqrt(3), 1.0 / np.sqrt(3)]
        weights = [1.0, 1.0]
    elif n_gauss == 3:
        xi_points = [-np.sqrt(3 / 5), 0, np.sqrt(3 / 5)]
        weights = [5 / 9, 8 / 9, 5 / 9]
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
            raise ValueError(f'No material region found for element {elem} at center {elem_center}')
        k_elem = np.zeros((2, 2))
        for i in range(n_gauss):
            xi = xi_points[i]
            w = weights[i]
            dN1_dxi = -0.5
            dN2_dxi = 0.5
            jacobian = L / 2
            dN1_dx = dN1_dxi / jacobian
            dN2_dx = dN2_dxi / jacobian
            B = np.array([dN1_dx, dN2_dx])
            k_elem += B.reshape(2, 1) @ B.reshape(1, 2) * E * A * jacobian * w
        K_global[node1:node2 + 1, node1:node2 + 1] += k_elem
        f_elem = np.zeros(2)
        for i in range(n_gauss):
            xi = xi_points[i]
            w = weights[i]
            N1 = (1 - xi) / 2
            N2 = (1 + xi) / 2
            x_gauss = N1 * x1 + N2 * x2
            jacobian = L / 2
            f_elem[0] += N1 * body_force_fn(x_gauss) * A * jacobian * w
            f_elem[1] += N2 * body_force_fn(x_gauss) * A * jacobian * w
        F_global[node1:node2 + 1] += f_elem
    if neumann_bc_list is not None:
        for bc in neumann_bc_list:
            x_loc = bc['x_location']
            load_mag = bc['load_mag']
            node_idx = np.argmin(np.abs(node_coords - x_loc))
            F_global[node_idx] += load_mag
    dirichlet_nodes = []
    prescribed_displacements = []
    for bc in dirichlet_bc_list:
        x_loc = bc['x_location']
        u_prescribed = bc['u_prescribed']
        node_idx = np.argmin(np.abs(node_coords - x_loc))
        dirichlet_nodes.append(node_idx)
        prescribed_displacements.append(u_prescribed)
    dirichlet_nodes = np.array(dirichlet_nodes)
    prescribed_displacements = np.array(prescribed_displacements)
    sort_idx = np.argsort(dirichlet_nodes)
    dirichlet_nodes = dirichlet_nodes[sort_idx]
    prescribed_displacements = prescribed_displacements[sort_idx]
    free_nodes = np.setdiff1d(np.arange(n_nodes), dirichlet_nodes)
    K_ff = K_global[np.ix_(free_nodes, free_nodes)]
    K_fd = K_global[np.ix_(free_nodes, dirichlet_nodes)]
    F_f = F_global[free_nodes]
    u_free = np.linalg.solve(K_ff, F_f - K_fd @ prescribed_displacements)
    displacements = np.zeros(n_nodes)
    displacements[free_nodes] = u_free
    displacements[dirichlet_nodes] = prescribed_displacements
    reactions = K_global[dirichlet_nodes] @ displacements - F_global[dirichlet_nodes]
    return {'displacements': displacements, 'reactions': reactions, 'node_coords': node_coords, 'reaction_nodes': dirichlet_nodes}