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
    if n_gauss == 1:
        gauss_pts = np.array([0.0])
        gauss_wts = np.array([2.0])
    elif n_gauss == 2:
        gauss_pts = np.array([-1.0 / np.sqrt(3.0), 1.0 / np.sqrt(3.0)])
        gauss_wts = np.array([1.0, 1.0])
    elif n_gauss == 3:
        gauss_pts = np.array([-np.sqrt(3.0 / 5.0), 0.0, np.sqrt(3.0 / 5.0)])
        gauss_wts = np.array([5.0 / 9.0, 8.0 / 9.0, 5.0 / 9.0])
    else:
        raise ValueError('n_gauss must be 1, 2, or 3')
    for e in range(num_elements):
        node_indices = np.array([e, e + 1])
        elem_coords = node_coords[node_indices]
        L_e = elem_coords[1] - elem_coords[0]
        x_mid = (elem_coords[0] + elem_coords[1]) / 2.0
        (E, A) = (None, None)
        for region in material_regions:
            if region['coord_min'] <= x_mid <= region['coord_max']:
                E = region['E']
                A = region['A']
                break
        if E is None or A is None:
            raise ValueError(f'No material properties found for element at midpoint {x_mid}')
        k_e = E * A / L_e * np.array([[1.0, -1.0], [-1.0, 1.0]])
        K_global[np.ix_(node_indices, node_indices)] += k_e
        f_e = np.zeros(2)
        J = L_e / 2.0
        for i in range(n_gauss):
            xi = gauss_pts[i]
            w = gauss_wts[i]
            N1 = 0.5 * (1.0 - xi)
            N2 = 0.5 * (1.0 + xi)
            x_gauss = N1 * elem_coords[0] + N2 * elem_coords[1]
            f_val = body_force_fn(x_gauss)
            f_e[0] += N1 * f_val * A * J * w
            f_e[1] += N2 * f_val * A * J * w
        F_global[node_indices] += f_e
    K_orig = K_global.copy()
    if neumann_bc_list is not None:
        for bc in neumann_bc_list:
            node_idx = np.argmin(np.abs(node_coords - bc['x_location']))
            F_global[node_idx] += bc['load_mag']
    dirichlet_nodes_set = set()
    dirichlet_values = {}
    for bc in dirichlet_bc_list:
        node_idx = np.argmin(np.abs(node_coords - bc['x_location']))
        dirichlet_nodes_set.add(node_idx)
        dirichlet_values[node_idx] = bc['u_prescribed']
    dirichlet_nodes = np.array(sorted(list(dirichlet_nodes_set)), dtype=int)
    all_nodes = np.arange(n_nodes)
    free_nodes = np.setdiff1d(all_nodes, dirichlet_nodes, assume_unique=True)
    U_d = np.array([dirichlet_values[i] for i in dirichlet_nodes])
    if len(free_nodes) > 0:
        K_ff = K_global[np.ix_(free_nodes, free_nodes)]
        K_fd = K_global[np.ix_(free_nodes, dirichlet_nodes)]
        F_f = F_global[free_nodes]
        F_f_mod = F_f - K_fd @ U_d
        U_f = np.linalg.solve(K_ff, F_f_mod)
    else:
        U_f = np.array([])
    displacements = np.zeros(n_nodes)
    if len(free_nodes) > 0:
        displacements[free_nodes] = U_f
    if len(dirichlet_nodes) > 0:
        displacements[dirichlet_nodes] = U_d
    if len(dirichlet_nodes) > 0:
        K_df = K_orig[np.ix_(dirichlet_nodes, free_nodes)]
        K_dd = K_orig[np.ix_(dirichlet_nodes, dirichlet_nodes)]
        F_d = F_global[dirichlet_nodes]
        if len(free_nodes) > 0:
            reactions = K_df @ U_f + K_dd @ U_d - F_d
        else:
            reactions = K_dd @ U_d - F_d
    else:
        reactions = np.array([])
    return {'displacements': displacements, 'reactions': reactions, 'node_coords': node_coords, 'reaction_nodes': dirichlet_nodes}