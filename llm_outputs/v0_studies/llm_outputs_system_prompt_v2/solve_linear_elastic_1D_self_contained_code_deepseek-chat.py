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
    element_length = (x_max - x_min) / num_elements
    (gauss_points, gauss_weights) = {1: (np.array([0.0]), np.array([2.0])), 2: (np.array([-1 / np.sqrt(3), 1 / np.sqrt(3)]), np.array([1.0, 1.0])), 3: (np.array([-np.sqrt(3 / 5), 0.0, np.sqrt(3 / 5)]), np.array([5 / 9, 8 / 9, 5 / 9]))}[n_gauss]
    K_global = np.zeros((n_nodes, n_nodes))
    F_global = np.zeros(n_nodes)
    for elem in range(num_elements):
        x1 = node_coords[elem]
        x2 = node_coords[elem + 1]
        le = x2 - x1
        E_val = None
        A_val = None
        for region in material_regions:
            if region['coord_min'] <= x1 and x2 <= region['coord_max']:
                E_val = region['E']
                A_val = region['A']
                break
        if E_val is None:
            raise ValueError(f'No material region found for element {elem} with nodes at {x1}, {x2}')
        k_elem = np.zeros((2, 2))
        f_elem = np.zeros(2)
        for i in range(n_gauss):
            xi = gauss_points[i]
            w = gauss_weights[i]
            N = np.array([0.5 * (1 - xi), 0.5 * (1 + xi)])
            dN_dxi = np.array([-0.5, 0.5])
            dx_dxi = le / 2
            dN_dx = dN_dxi / dx_dxi
            B = dN_dx
            x_phys = 0.5 * (1 - xi) * x1 + 0.5 * (1 + xi) * x2
            k_elem += B.reshape(2, 1) @ B.reshape(1, 2) * (E_val * A_val) * w * dx_dxi
            f_elem += N * body_force_fn(x_phys) * w * dx_dxi
        K_global[elem:elem + 2, elem:elem + 2] += k_elem
        F_global[elem:elem + 2] += f_elem
    if neumann_bc_list is not None:
        for bc in neumann_bc_list:
            x_loc = bc['x_location']
            load = bc['load_mag']
            node_idx = np.argmin(np.abs(node_coords - x_loc))
            F_global[node_idx] += load
    dirichlet_nodes = []
    prescribed_values = []
    for bc in dirichlet_bc_list:
        x_loc = bc['x_location']
        u_pres = bc['u_prescribed']
        node_idx = np.argmin(np.abs(node_coords - x_loc))
        dirichlet_nodes.append(node_idx)
        prescribed_values.append(u_pres)
    free_nodes = [i for i in range(n_nodes) if i not in dirichlet_nodes]
    if free_nodes:
        K_ff = K_global[np.ix_(free_nodes, free_nodes)]
        K_fd = K_global[np.ix_(free_nodes, dirichlet_nodes)]
        F_f = F_global[free_nodes]
        u_d = np.array(prescribed_values)
        F_f_modified = F_f - K_fd @ u_d
        u_f = np.linalg.solve(K_ff, F_f_modified)
        displacements = np.zeros(n_nodes)
        displacements[free_nodes] = u_f
        displacements[dirichlet_nodes] = u_d
    else:
        displacements = np.array(prescribed_values)
    reactions = K_global[dirichlet_nodes, :] @ displacements - F_global[dirichlet_nodes]
    return {'displacements': displacements, 'reactions': reactions, 'node_coords': node_coords, 'reaction_nodes': np.array(dirichlet_nodes)}