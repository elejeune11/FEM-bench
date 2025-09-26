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
    (gauss_points, gauss_weights) = {1: (np.array([0.0]), np.array([2.0])), 2: (np.array([-1 / np.sqrt(3), 1 / np.sqrt(3)]), np.array([1.0, 1.0])), 3: (np.array([-np.sqrt(3 / 5), 0.0, np.sqrt(3 / 5)]), np.array([5 / 9, 8 / 9, 5 / 9]))}[n_gauss]
    K_global = np.zeros((n_nodes, n_nodes))
    F_global = np.zeros(n_nodes)
    for i in range(num_elements):
        (x1, x2) = (node_coords[i], node_coords[i + 1])
        L = x2 - x1
        center = (x1 + x2) / 2
        for region in material_regions:
            if region['coord_min'] <= center <= region['coord_max']:
                E = region['E']
                A = region['A']
                break
        k_e = E * A / L * np.array([[1, -1], [-1, 1]])
        K_global[i:i + 2, i:i + 2] += k_e
        for (gp, w) in zip(gauss_points, gauss_weights):
            xi = gp
            N = np.array([(1 - xi) / 2, (1 + xi) / 2])
            dN_dxi = np.array([-0.5, 0.5])
            x = N[0] * x1 + N[1] * x2
            jacobian = L / 2
            b = body_force_fn(x)
            F_global[i:i + 2] += N * b * w * jacobian
    if neumann_bc_list:
        for bc in neumann_bc_list:
            node_idx = np.argmin(np.abs(node_coords - bc['x_location']))
            F_global[node_idx] += bc['load_mag']
    dirichlet_nodes = []
    prescribed_values = []
    for bc in dirichlet_bc_list:
        node_idx = np.argmin(np.abs(node_coords - bc['x_location']))
        dirichlet_nodes.append(node_idx)
        prescribed_values.append(bc['u_prescribed'])
    free_nodes = [i for i in range(n_nodes) if i not in dirichlet_nodes]
    if free_nodes:
        K_ff = K_global[np.ix_(free_nodes, free_nodes)]
        K_fd = K_global[np.ix_(free_nodes, dirichlet_nodes)]
        F_f = F_global[free_nodes]
        u_prescribed = np.array(prescribed_values)
        u_free = np.linalg.solve(K_ff, F_f - K_fd @ u_prescribed)
        displacements = np.zeros(n_nodes)
        displacements[free_nodes] = u_free
        displacements[dirichlet_nodes] = u_prescribed
    else:
        displacements = np.array(prescribed_values)
    reactions = K_global[dirichlet_nodes] @ displacements - F_global[dirichlet_nodes]
    return {'displacements': displacements, 'reactions': reactions, 'node_coords': node_coords, 'reaction_nodes': np.array(dirichlet_nodes)}