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
    element_nodes = np.array([[i, i + 1] for i in range(num_elements)])
    K = np.zeros((n_nodes, n_nodes))
    F = np.zeros(n_nodes)
    if n_gauss == 1:
        gauss_points = np.array([0.0])
        gauss_weights = np.array([2.0])
    elif n_gauss == 2:
        gauss_points = np.array([-1.0 / np.sqrt(3.0), 1.0 / np.sqrt(3.0)])
        gauss_weights = np.array([1.0, 1.0])
    elif n_gauss == 3:
        gauss_points = np.array([-np.sqrt(3.0 / 5.0), 0.0, np.sqrt(3.0 / 5.0)])
        gauss_weights = np.array([5.0 / 9.0, 8.0 / 9.0, 5.0 / 9.0])
    else:
        raise ValueError('n_gauss must be 1, 2, or 3')
    for e in range(num_elements):
        node_indices = element_nodes[e]
        (x_i, x_j) = node_coords[node_indices]
        L_e = x_j - x_i
        x_mid = (x_i + x_j) / 2.0
        (E, A) = (None, None)
        for region in material_regions:
            if region['coord_min'] <= x_mid <= region['coord_max']:
                E = region['E']
                A = region['A']
                break
        if E is None or A is None:
            raise ValueError(f'No material defined for element at x={x_mid}')
        k_e = E * A / L_e * np.array([[1.0, -1.0], [-1.0, 1.0]])
        f_e = np.zeros(2)
        jacobian = L_e / 2.0
        for (gp, gw) in zip(gauss_points, gauss_weights):
            N_i = 0.5 * (1.0 - gp)
            N_j = 0.5 * (1.0 + gp)
            x_gauss = N_i * x_i + N_j * x_j
            body_force_val = body_force_fn(x_gauss)
            f_e[0] += N_i * body_force_val * A * jacobian * gw
            f_e[1] += N_j * body_force_val * A * jacobian * gw
        K[np.ix_(node_indices, node_indices)] += k_e
        F[node_indices] += f_e
    if neumann_bc_list:
        for bc in neumann_bc_list:
            node_idx = np.argmin(np.abs(node_coords - bc['x_location']))
            F[node_idx] += bc['load_mag']
    bc_map = {}
    for bc in dirichlet_bc_list:
        node_idx = np.argmin(np.abs(node_coords - bc['x_location']))
        bc_map[node_idx] = bc['u_prescribed']
    dirichlet_nodes = np.array(sorted(bc_map.keys()), dtype=int)
    u_d = np.array([bc_map[n] for n in dirichlet_nodes])
    all_nodes = np.arange(n_nodes)
    unknown_nodes = np.setdiff1d(all_nodes, dirichlet_nodes)
    if unknown_nodes.size > 0:
        K_uu = K[np.ix_(unknown_nodes, unknown_nodes)]
        K_ud = K[np.ix_(unknown_nodes, dirichlet_nodes)]
        F_u = F[unknown_nodes]
        F_reduced = F_u - K_ud @ u_d
        u_u = np.linalg.solve(K_uu, F_reduced)
    else:
        u_u = np.array([])
    displacements = np.zeros(n_nodes)
    if unknown_nodes.size > 0:
        displacements[unknown_nodes] = u_u
    if dirichlet_nodes.size > 0:
        displacements[dirichlet_nodes] = u_d
    reactions = (K @ displacements - F)[dirichlet_nodes]
    return {'displacements': displacements, 'reactions': reactions, 'node_coords': node_coords, 'reaction_nodes': dirichlet_nodes}