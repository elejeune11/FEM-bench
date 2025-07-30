def solve_linear_elastic_1D_self_contained(x_min: float, x_max: float, num_elements: int, material_regions: List[Dict[str, float]], body_force_fn: Callable[[float], float], dirichlet_bc_list: List[Dict[str, float]], neumann_bc_list: Optional[List[Dict[str, float]]], n_gauss: int) -> Dict[str, np.ndarray]:
    """
    Solve a D linear elastic finite element problem with integrated meshing.
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
    num_nodes = len(node_coords)
    elements = np.array([[i, i + 1] for i in range(num_elements)])
    K_global = np.zeros((num_nodes, num_nodes))
    F_global = np.zeros(num_nodes)
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
    for elem in elements:
        (x1, x2) = (node_coords[elem[0]], node_coords[elem[1]])
        length = x2 - x1
        jacobian = length / 2
        (E, A) = (None, None)
        for region in material_regions:
            if x1 >= region['coord_min'] and x2 <= region['coord_max']:
                E = region['E']
                A = region['A']
                break
        if E is None or A is None:
            raise ValueError('Element not covered by any material region')
        k_elem = E * A / length * np.array([[1, -1], [-1, 1]])
        for (i, node_i) in enumerate(elem):
            for (j, node_j) in enumerate(elem):
                K_global[node_i, node_j] += k_elem[i, j]
        for (gp, weight) in zip(gauss_points, gauss_weights):
            xi = gp
            N1 = 0.5 * (1 - xi)
            N2 = 0.5 * (1 + xi)
            x = N1 * x1 + N2 * x2
            f = body_force_fn(x)
            F_global[elem[0]] += f * N1 * weight * jacobian
            F_global[elem[1]] += f * N2 * weight * jacobian
    if neumann_bc_list is not None:
        for bc in neumann_bc_list:
            x_loc = bc['x_location']
            load = bc['load_mag']
            node_idx = np.argmin(np.abs(node_coords - x_loc))
            F_global[node_idx] += load
    dirichlet_nodes = []
    u_prescribed = []
    for bc in dirichlet_bc_list:
        x_loc = bc['x_location']
        u_pres = bc['u_prescribed']
        node_idx = np.argmin(np.abs(node_coords - x_loc))
        dirichlet_nodes.append(node_idx)
        u_prescribed.append(u_pres)
    u_solution = np.zeros(num_nodes)
    for (i, node) in enumerate(dirichlet_nodes):
        u_solution[node] = u_prescribed[i]
        F_global -= K_global[:, node] * u_prescribed[i]
    free_nodes = [i for i in range(num_nodes) if i not in dirichlet_nodes]
    K_reduced = K_global[np.ix_(free_nodes, free_nodes)]
    F_reduced = F_global[free_nodes]
    u_reduced = np.linalg.solve(K_reduced, F_reduced)
    u_solution[free_nodes] = u_reduced
    reactions = K_global[dirichlet_nodes] @ u_solution - F_global[dirichlet_nodes]
    return {'displacements': u_solution, 'reactions': reactions, 'node_coords': node_coords, 'reaction_nodes': np.array(dirichlet_nodes)}