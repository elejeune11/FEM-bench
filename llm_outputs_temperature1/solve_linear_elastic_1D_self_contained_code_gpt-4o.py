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
    gauss_points = {1: [0.0], 2: [-1 / np.sqrt(3), 1 / np.sqrt(3)], 3: [-np.sqrt(3 / 5), 0.0, np.sqrt(3 / 5)]}
    gauss_weights = {1: [2.0], 2: [1.0, 1.0], 3: [5 / 9, 8 / 9, 5 / 9]}
    for i in range(num_elements):
        (x1, x2) = (node_coords[i], node_coords[i + 1])
        L = x2 - x1
        for region in material_regions:
            if region['coord_min'] <= x1 < region['coord_max']:
                E = region['E']
                A = region['A']
                break
        k_local = E * A / L * np.array([[1, -1], [-1, 1]])
        f_local = np.zeros(2)
        for (gp, gw) in zip(gauss_points[n_gauss], gauss_weights[n_gauss]):
            xi = (x2 - x1) / 2 * gp + (x2 + x1) / 2
            N = np.array([1 - gp, 1 + gp]) / 2
            f_local += gw * body_force_fn(xi) * N * L / 2
        K_global[i:i + 2, i:i + 2] += k_local
        F_global[i:i + 2] += f_local
    if neumann_bc_list:
        for bc in neumann_bc_list:
            x_loc = bc['x_location']
            load_mag = bc['load_mag']
            node_index = np.argmin(np.abs(node_coords - x_loc))
            F_global[node_index] += load_mag
    u = np.zeros(n_nodes)
    reaction_nodes = []
    for bc in dirichlet_bc_list:
        x_loc = bc['x_location']
        u_prescribed = bc['u_prescribed']
        node_index = np.argmin(np.abs(node_coords - x_loc))
        reaction_nodes.append(node_index)
        u[node_index] = u_prescribed
    free_nodes = np.setdiff1d(np.arange(n_nodes), reaction_nodes)
    K_ff = K_global[np.ix_(free_nodes, free_nodes)]
    K_fr = K_global[np.ix_(free_nodes, reaction_nodes)]
    F_f = F_global[free_nodes]
    u_free = np.linalg.solve(K_ff, F_f - K_fr @ u[reaction_nodes])
    u[free_nodes] = u_free
    reactions = K_global @ u - F_global
    reaction_forces = reactions[reaction_nodes]
    return {'displacements': u, 'reactions': reaction_forces, 'node_coords': node_coords, 'reaction_nodes': np.array(reaction_nodes)}