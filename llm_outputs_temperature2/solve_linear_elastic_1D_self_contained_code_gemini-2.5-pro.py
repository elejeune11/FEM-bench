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
        raise ValueError('n_gauss must be 1, 2, or 3.')
    for e in range(num_elements):
        node_indices = np.array([e, e + 1])
        (x_i, x_j) = node_coords[node_indices]
        L_e = x_j - x_i
        x_mid = (x_i + x_j) / 2.0
        (E, A) = (None, None)
        for region in material_regions:
            if region['coord_min'] - 1e-09 <= x_mid <= region['coord_max'] + 1e-09:
                E = region['E']
                A = region['A']
                break
        if E is None or A is None:
            raise ValueError(f'No material properties found for element centered at x={x_mid}')
        k_e = E * A / L_e * np.array([[1.0, -1.0], [-1.0, 1.0]])
        K[np.ix_(node_indices, node_indices)] += k_e
        f_e_body = np.zeros(2)
        J = L_e / 2.0
        for i in range(n_gauss):
            xi = gauss_points[i]
            w = gauss_weights[i]
            N1 = (1.0 - xi) / 2.0
            N2 = (1.0 + xi) / 2.0
            N_g = np.array([N1, N2])
            x_g = N1 * x_i + N2 * x_j
            f_val = body_force_fn(x_g)
            f_e_body += w * N_g * f_val * J
        F[node_indices] += f_e_body
    if neumann_bc_list is not None:
        for bc in neumann_bc_list:
            x_loc = bc['x_location']
            load = bc['load_mag']
            node_idx = np.argmin(np.abs(node_coords - x_loc))
            F[node_idx] += load
    prescribed_map = {np.argmin(np.abs(node_coords - bc['x_location'])): bc['u_prescribed'] for bc in dirichlet_bc_list}
    dirichlet_nodes = np.array(sorted(prescribed_map.keys()), dtype=int)
    U_d = np.array([prescribed_map[i] for i in dirichlet_nodes])
    all_nodes = np.arange(n_nodes)
    free_nodes = np.setdiff1d(all_nodes, dirichlet_nodes, assume_unique=True)
    K_ff = K[np.ix_(free_nodes, free_nodes)]
    K_fd = K[np.ix_(free_nodes, dirichlet_nodes)]
    F_f = F[free_nodes]
    F_f_modified = F_f - K_fd @ U_d
    U_f = np.linalg.solve(K_ff, F_f_modified)
    displacements = np.zeros(n_nodes)
    displacements[free_nodes] = U_f
    displacements[dirichlet_nodes] = U_d
    K_df = K[np.ix_(dirichlet_nodes, free_nodes)]
    K_dd = K[np.ix_(dirichlet_nodes, dirichlet_nodes)]
    F_d = F[dirichlet_nodes]
    reactions = K_df @ U_f + K_dd @ U_d - F_d
    return {'displacements': displacements, 'reactions': reactions, 'node_coords': node_coords, 'reaction_nodes': dirichlet_nodes}