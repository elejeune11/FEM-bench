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
    K_global = np.zeros((n_nodes, n_nodes))
    F_global = np.zeros(n_nodes)
    gauss_points = {1: [0.0], 2: [-1 / np.sqrt(3), 1 / np.sqrt(3)], 3: [-np.sqrt(3 / 5), 0.0, np.sqrt(3 / 5)]}
    gauss_weights = {1: [2.0], 2: [1.0, 1.0], 3: [5 / 9, 8 / 9, 5 / 9]}
    xi_vals = gauss_points[n_gauss]
    w_vals = gauss_weights[n_gauss]
    for elem in range(num_elements):
        node_left = elem
        node_right = elem + 1
        x_left = node_coords[node_left]
        x_right = node_coords[node_right]
        L = x_right - x_left
        E_val = 1.0
        A_val = 1.0
        for region in material_regions:
            if region['coord_min'] <= x_left <= region['coord_max'] and region['coord_min'] <= x_right <= region['coord_max']:
                E_val = region['E']
                A_val = region['A']
                break
        k_elem = np.zeros((2, 2))
        f_elem = np.zeros(2)
        for i in range(n_gauss):
            xi = xi_vals[i]
            w = w_vals[i]
            N1 = 0.5 * (1 - xi)
            N2 = 0.5 * (1 + xi)
            dN1_dxi = -0.5
            dN2_dxi = 0.5
            dx_dxi = 0.5 * L
            dxi_dx = 2.0 / L
            B = np.array([dN1_dxi * dxi_dx, dN2_dxi * dxi_dx])
            x_gauss = N1 * x_left + N2 * x_right
            k_elem += E_val * A_val * np.outer(B, B) * dx_dxi * w
            f_elem += body_force_fn(x_gauss) * np.array([N1, N2]) * dx_dxi * w
        K_global[node_left:node_right + 1:1, node_left:node_right + 1:1] += k_elem
        F_global[node_left:node_right + 1:1] += f_elem
    if neumann_bc_list is not None:
        for bc in neumann_bc_list:
            x_loc = bc['x_location']
            load_mag = bc['load_mag']
            node_idx = np.argmin(np.abs(node_coords - x_loc))
            F_global[node_idx] += load_mag
    dirichlet_nodes = []
    prescribed_values = []
    for bc in dirichlet_bc_list:
        x_loc = bc['x_location']
        u_prescribed = bc['u_prescribed']
        node_idx = np.argmin(np.abs(node_coords - x_loc))
        dirichlet_nodes.append(node_idx)
        prescribed_values.append(u_prescribed)
    free_nodes = [i for i in range(n_nodes) if i not in dirichlet_nodes]
    K_ff = K_global[np.ix_(free_nodes, free_nodes)]
    K_fd = K_global[np.ix_(free_nodes, dirichlet_nodes)]
    F_f = F_global[free_nodes]
    u_prescribed_arr = np.array(prescribed_values)
    F_f_modified = F_f - K_fd @ u_prescribed_arr
    u_free = np.linalg.solve(K_ff, F_f_modified)
    displacements = np.zeros(n_nodes)
    displacements[free_nodes] = u_free
    displacements[dirichlet_nodes] = u_prescribed_arr
    reactions = K_global[dirichlet_nodes, :] @ displacements - F_global[dirichlet_nodes]
    return {'displacements': displacements, 'reactions': reactions, 'node_coords': node_coords, 'reaction_nodes': np.array(dirichlet_nodes)}