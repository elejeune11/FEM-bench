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
    K = np.zeros((n_nodes, n_nodes))
    F = np.zeros(n_nodes)
    gauss_points = {1: [0.0], 2: [-0.5773502691896257, 0.5773502691896257], 3: [-0.7745966692414834, 0.0, 0.7745966692414834]}
    gauss_weights = {1: [2.0], 2: [1.0, 1.0], 3: [0.5555555555555556, 0.8888888888888888, 0.5555555555555556]}
    for i in range(num_elements):
        h = node_coords[i + 1] - node_coords[i]
        x_mid = (node_coords[i + 1] + node_coords[i]) / 2
        for region in material_regions:
            if region['coord_min'] <= x_mid <= region['coord_max']:
                E = region['E']
                A = region['A']
                break
        ke = E * A / h * np.array([[1, -1], [-1, 1]])
        K[i:i + 2, i:i + 2] += ke
        for (j, xi) in enumerate(gauss_points[n_gauss]):
            x_gauss = x_mid + xi * h / 2
            w = gauss_weights[n_gauss][j]
            f = body_force_fn(x_gauss)
            fe = f * A * w * h / 2 * np.array([1, 1])
            F[i:i + 2] += fe
    if neumann_bc_list:
        for neumann_bc in neumann_bc_list:
            node_index = np.where(np.isclose(node_coords, neumann_bc['x_location']))[0][0]
            F[node_index] += neumann_bc['load_mag']
    dirichlet_indices = []
    dirichlet_values = []
    for dirichlet_bc in dirichlet_bc_list:
        node_index = np.where(np.isclose(node_coords, dirichlet_bc['x_location']))[0][0]
        dirichlet_indices.append(node_index)
        dirichlet_values.append(dirichlet_bc['u_prescribed'])
    free_indices = np.setdiff1d(np.arange(n_nodes), dirichlet_indices)
    K_ff = K[np.ix_(free_indices, free_indices)]
    F_f = F[free_indices]
    u_f = np.linalg.solve(K_ff, F_f)
    u = np.zeros(n_nodes)
    u[free_indices] = u_f
    u[dirichlet_indices] = dirichlet_values
    reactions = K[dirichlet_indices, :] @ u - F[dirichlet_indices]
    return {'displacements': u, 'reactions': reactions, 'node_coords': node_coords, 'reaction_nodes': np.array(dirichlet_indices)}