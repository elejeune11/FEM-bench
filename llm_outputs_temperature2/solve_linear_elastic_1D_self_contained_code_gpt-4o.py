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
    for elem in range(num_elements):
        x1 = node_coords[elem]
        x2 = node_coords[elem + 1]
        length = x2 - x1
        E = A = None
        for region in material_regions:
            if region['coord_min'] <= x1 < region['coord_max']:
                E = region['E']
                A = region['A']
                break
        if E is None or A is None:
            raise ValueError('Material properties not defined for element starting at x = {}'.format(x1))
        k_local = E * A / length * np.array([[1, -1], [-1, 1]])
        f_local = np.zeros(2)
        for (gp, gw) in zip(gauss_points, gauss_weights):
            xi = (x2 - x1) / 2 * gp + (x1 + x2) / 2
            N1 = 0.5 * (1 - gp)
            N2 = 0.5 * (1 + gp)
            f_local[0] += gw * body_force_fn(xi) * N1 * length / 2
            f_local[1] += gw * body_force_fn(xi) * N2 * length / 2
        indices = [elem, elem + 1]
        for i in range(2):
            F_global[indices[i]] += f_local[i]
            for j in range(2):
                K_global[indices[i], indices[j]] += k_local[i, j]
    if neumann_bc_list:
        for bc in neumann_bc_list:
            x_loc = bc['x_location']
            load_mag = bc['load_mag']
            node_index = np.argmin(np.abs(node_coords - x_loc))
            F_global[node_index] += load_mag
    penalty = 1e+20
    reaction_nodes = []
    for bc in dirichlet_bc_list:
        x_loc = bc['x_location']
        u_prescribed = bc['u_prescribed']
        node_index = np.argmin(np.abs(node_coords - x_loc))
        reaction_nodes.append(node_index)
        K_global[node_index, node_index] += penalty
        F_global[node_index] += penalty * u_prescribed
    displacements = np.linalg.solve(K_global, F_global)
    reactions = np.array([penalty * (displacements[node_index] - bc['u_prescribed']) for (node_index, bc) in zip(reaction_nodes, dirichlet_bc_list)])
    return {'displacements': displacements, 'reactions': reactions, 'node_coords': node_coords, 'reaction_nodes': np.array(reaction_nodes)}