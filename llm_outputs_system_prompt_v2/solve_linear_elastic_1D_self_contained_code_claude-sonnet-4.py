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
    gauss_points = {1: (np.array([0.0]), np.array([2.0])), 2: (np.array([-1.0 / np.sqrt(3), 1.0 / np.sqrt(3)]), np.array([1.0, 1.0])), 3: (np.array([-np.sqrt(3.0 / 5.0), 0.0, np.sqrt(3.0 / 5.0)]), np.array([5.0 / 9.0, 8.0 / 9.0, 5.0 / 9.0]))}
    (xi_gauss, w_gauss) = gauss_points[n_gauss]
    K_global = np.zeros((n_nodes, n_nodes))
    F_global = np.zeros(n_nodes)
    for elem in range(num_elements):
        (node1, node2) = (elem, elem + 1)
        (x1, x2) = (node_coords[node1], node_coords[node2])
        L_e = x2 - x1
        x_center = 0.5 * (x1 + x2)
        (E, A) = (None, None)
        for region in material_regions:
            if region['coord_min'] <= x_center <= region['coord_max']:
                (E, A) = (region['E'], region['A'])
                break
        K_elem = np.zeros((2, 2))
        F_elem = np.zeros(2)
        for i in range(n_gauss):
            xi = xi_gauss[i]
            w = w_gauss[i]
            x_gauss = 0.5 * (x1 + x2) + 0.5 * xi * (x2 - x1)
            J = L_e / 2.0
            dN_dxi = np.array([-0.5, 0.5])
            dN_dx = dN_dxi / J
            K_elem += w * E * A * np.outer(dN_dx, dN_dx) * J
            N = np.array([0.5 * (1 - xi), 0.5 * (1 + xi)])
            f_val = body_force_fn(x_gauss)
            F_elem += w * f_val * A * N * J
        dof = [node1, node2]
        for i in range(2):
            for j in range(2):
                K_global[dof[i], dof[j]] += K_elem[i, j]
            F_global[dof[i]] += F_elem[i]
    if neumann_bc_list:
        for bc in neumann_bc_list:
            x_loc = bc['x_location']
            load = bc['load_mag']
            node_idx = np.argmin(np.abs(node_coords - x_loc))
            F_global[node_idx] += load
    dirichlet_nodes = []
    dirichlet_values = []
    for bc in dirichlet_bc_list:
        x_loc = bc['x_location']
        u_val = bc['u_prescribed']
        node_idx = np.argmin(np.abs(node_coords - x_loc))
        dirichlet_nodes.append(node_idx)
        dirichlet_values.append(u_val)
    dirichlet_nodes = np.array(dirichlet_nodes)
    dirichlet_values = np.array(dirichlet_values)
    free_nodes = np.setdiff1d(np.arange(n_nodes), dirichlet_nodes)
    K_ff = K_global[np.ix_(free_nodes, free_nodes)]
    K_fd = K_global[np.ix_(free_nodes, dirichlet_nodes)]
    F_f = F_global[free_nodes] - K_fd @ dirichlet_values
    u_free = np.linalg.solve(K_ff, F_f)
    u_full = np.zeros(n_nodes)
    u_full[free_nodes] = u_free
    u_full[dirichlet_nodes] = dirichlet_values
    K_df = K_global[np.ix_(dirichlet_nodes, free_nodes)]
    K_dd = K_global[np.ix_(dirichlet_nodes, dirichlet_nodes)]
    reactions = K_df @ u_free + K_dd @ dirichlet_values - F_global[dirichlet_nodes]
    return {'displacements': u_full, 'reactions': reactions, 'node_coords': node_coords, 'reaction_nodes': dirichlet_nodes}