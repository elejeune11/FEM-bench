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
    if n_gauss == 1:
        gauss_points = np.array([0.0])
        gauss_weights = np.array([2.0])
    elif n_gauss == 2:
        gauss_points = np.array([-1 / np.sqrt(3), 1 / np.sqrt(3)])
        gauss_weights = np.array([1.0, 1.0])
    elif n_gauss == 3:
        gauss_points = np.array([-np.sqrt(3 / 5), 0.0, np.sqrt(3 / 5)])
        gauss_weights = np.array([5 / 9, 8 / 9, 5 / 9])
    else:
        raise ValueError('n_gauss must be 1, 2, or 3.')
    node_coords = np.linspace(x_min, x_max, num_elements + 1)
    element_length = (x_max - x_min) / num_elements
    n_nodes = num_elements + 1
    K_global = np.zeros((n_nodes, n_nodes))
    F_global = np.zeros(n_nodes)
    for i in range(num_elements):
        x1 = node_coords[i]
        x2 = node_coords[i + 1]
        element_nodes = [i, i + 1]
        E = A = 0.0
        for region in material_regions:
            if x1 >= region['coord_min'] and x2 <= region['coord_max']:
                E = region['E']
                A = region['A']
                break
        K_element = np.array([[1, -1], [-1, 1]]) * (E * A / element_length)
        F_element = np.zeros(2)
        for (gp, gw) in zip(gauss_points, gauss_weights):
            xi = (x2 - x1) / 2 * gp + (x2 + x1) / 2
            N = np.array([1 - gp, 1 + gp]) / 2
            f = body_force_fn(xi)
            F_element += N * f * gw * element_length / 2
        for a in range(2):
            F_global[element_nodes[a]] += F_element[a]
            for b in range(2):
                K_global[element_nodes[a], element_nodes[b]] += K_element[a, b]
    if neumann_bc_list:
        for bc in neumann_bc_list:
            x_loc = bc['x_location']
            load_mag = bc['load_mag']
            node_index = np.argmin(np.abs(node_coords - x_loc))
            F_global[node_index] += load_mag
    reaction_nodes = []
    for bc in dirichlet_bc_list:
        x_loc = bc['x_location']
        u_prescribed = bc['u_prescribed']
        node_index = np.argmin(np.abs(node_coords - x_loc))
        reaction_nodes.append(node_index)
        K_global[node_index, :] = 0
        K_global[node_index, node_index] = 1
        F_global[node_index] = u_prescribed
    displacements = np.linalg.solve(K_global, F_global)
    reactions = np.array([K_global[node, :] @ displacements - F_global[node] for node in reaction_nodes])
    return {'displacements': displacements, 'reactions': reactions, 'node_coords': node_coords, 'reaction_nodes': np.array(reaction_nodes)}