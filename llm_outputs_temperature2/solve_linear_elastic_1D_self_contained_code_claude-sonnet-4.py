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
    if n_gauss == 1:
        gauss_points = np.array([0.0])
        gauss_weights = np.array([2.0])
    elif n_gauss == 2:
        gauss_points = np.array([-1.0 / np.sqrt(3), 1.0 / np.sqrt(3)])
        gauss_weights = np.array([1.0, 1.0])
    elif n_gauss == 3:
        gauss_points = np.array([-np.sqrt(3.0 / 5.0), 0.0, np.sqrt(3.0 / 5.0)])
        gauss_weights = np.array([5.0 / 9.0, 8.0 / 9.0, 5.0 / 9.0])
    K_global = np.zeros((n_nodes, n_nodes))
    F_global = np.zeros(n_nodes)
    for elem in range(num_elements):
        x1 = node_coords[elem]
        x2 = node_coords[elem + 1]
        L = x2 - x1
        elem_center = (x1 + x2) / 2
        E = None
        A = None
        for region in material_regions:
            if region['coord_min'] <= elem_center <= region['coord_max']:
                E = region['E']
                A = region['A']
                break
        K_elem = np.zeros((2, 2))
        F_elem = np.zeros(2)
        for gp in range(n_gauss):
            xi = gauss_points[gp]
            w = gauss_weights[gp]
            dN_dxi = np.array([-1.0, 1.0])
            J = L / 2.0
            dN_dx = dN_dxi / J
            x = x1 + (xi + 1) * L / 2
            N = np.array([(1 - xi) / 2, (1 + xi) / 2])
            K_elem += E * A * np.outer(dN_dx, dN_dx) * J * w
            f_val = body_force_fn(x)
            F_elem += f_val * A * N * J * w
        nodes = [elem, elem + 1]
        for i in range(2):
            for j in range(2):
                K_global[nodes[i], nodes[j]] += K_elem[i, j]
            F_global[nodes[i]] += F_elem[i]
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
    K_reduced = K_global[np.ix_(free_nodes, free_nodes)]
    F_reduced = F_global[free_nodes]
    if len(dirichlet_nodes) > 0:
        F_reduced -= K_global[np.ix_(free_nodes, dirichlet_nodes)] @ dirichlet_values
    u_free = np.linalg.solve(K_reduced, F_reduced)
    displacements = np.zeros(n_nodes)
    displacements[free_nodes] = u_free
    displacements[dirichlet_nodes] = dirichlet_values
    reactions = K_global[dirichlet_nodes, :] @ displacements - F_global[dirichlet_nodes]
    return {'displacements': displacements, 'reactions': reactions, 'node_coords': node_coords, 'reaction_nodes': dirichlet_nodes}