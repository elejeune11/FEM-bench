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
    h = (x_max - x_min) / num_elements
    node_coords = np.linspace(x_min, x_max, num_elements + 1)
    n_nodes = len(node_coords)
    E_list = []
    A_list = []
    for element in range(num_elements):
        x_start = node_coords[element]
        x_end = node_coords[element + 1]
        for region in material_regions:
            if x_start >= region['coord_min'] and x_end <= region['coord_max']:
                E_list.append(region['E'])
                A_list.append(region['A'])
                break
    K = np.zeros((n_nodes, n_nodes))
    for element in range(num_elements):
        x1 = node_coords[element]
        x2 = node_coords[element + 1]
        E = E_list[element]
        A = A_list[element]
        k_element = E * A / h * np.array([[1, -1], [-1, 1]])
        K[element:element + 2, element:element + 2] += k_element
    f = np.zeros(n_nodes)
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
        raise ValueError('n_gauss must be 1, 2, or 3.')
    for element in range(num_elements):
        x1 = node_coords[element]
        x2 = node_coords[element + 1]
        for (gp, w) in zip(gauss_points, gauss_weights):
            x_gp = x1 + (x2 - x1) * (gp + 1) / 2
            f_gp = body_force_fn(x_gp)
            f[element] += f_gp * w * (x2 - x1) / 2
            f[element + 1] += f_gp * w * (x2 - x1) / 2
    u_prescribed = np.zeros(n_nodes)
    reaction_nodes = []
    for bc in dirichlet_bc_list:
        node_index = np.argmin(np.abs(node_coords - bc['x_location']))
        u_prescribed[node_index] = bc['u_prescribed']
        reaction_nodes.append(node_index)
    if neumann_bc_list:
        for bc in neumann_bc_list:
            node_index = np.argmin(np.abs(node_coords - bc['x_location']))
            f[node_index] += bc['load_mag']
    K_modified = K.copy()
    f_modified = f.copy()
    for i in reaction_nodes:
        K_modified[i, :] = 0
        K_modified[i, i] = 1
        f_modified[i] = u_prescribed[i]
    u = np.linalg.solve(K_modified, f_modified)
    reactions = np.dot(K, u) - f
    reactions = reactions[reaction_nodes]
    return {'displacements': u, 'reactions': reactions, 'node_coords': node_coords, 'reaction_nodes': np.array(reaction_nodes)}