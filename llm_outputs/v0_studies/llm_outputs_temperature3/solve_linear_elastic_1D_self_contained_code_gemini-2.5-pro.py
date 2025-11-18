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
        gp = 1.0 / np.sqrt(3.0)
        gauss_points = np.array([-gp, gp])
        gauss_weights = np.array([1.0, 1.0])
    elif n_gauss == 3:
        gp = np.sqrt(3.0 / 5.0)
        gauss_points = np.array([-gp, 0.0, gp])
        gauss_weights = np.array([5.0 / 9.0, 8.0 / 9.0, 5.0 / 9.0])
    else:
        raise ValueError('n_gauss must be 1, 2, or 3')
    n_nodes = num_elements + 1
    node_coords = np.linspace(x_min, x_max, n_nodes)
    K = np.zeros((n_nodes, n_nodes))
    F = np.zeros(n_nodes)
    for e in range(num_elements):
        node_indices = np.array([e, e + 1])
        elem_coords = node_coords[node_indices]
        L_e = elem_coords[1] - elem_coords[0]
        J = L_e / 2.0
        elem_mid_coord = np.mean(elem_coords)
        (E, A) = (None, None)
        for region in material_regions:
            if region['coord_min'] <= elem_mid_coord < region['coord_max'] or (np.isclose(elem_mid_coord, region['coord_max']) and np.isclose(elem_mid_coord, x_max)):
                E = region['E']
                A = region['A']
                break
        if E is None or A is None:
            for region in material_regions:
                if region['coord_min'] <= elem_mid_coord <= region['coord_max']:
                    E = region['E']
                    A = region['A']
                    break
        if E is None or A is None:
            raise ValueError(f'No material properties found for element at x={elem_mid_coord}')
        k_e = np.zeros((2, 2))
        f_e = np.zeros(2)
        for (gp, w) in zip(gauss_points, gauss_weights):
            N = np.array([(1.0 - gp) / 2.0, (1.0 + gp) / 2.0])
            dN_dxi = np.array([-0.5, 0.5])
            B = dN_dxi / J
            k_e += np.outer(B, B) * E * A * w * J
            x_gp = N @ elem_coords
            f_e += N * body_force_fn(x_gp) * w * J
        K[np.ix_(node_indices, node_indices)] += k_e
        F[node_indices] += f_e
    if neumann_bc_list is not None:
        for bc in neumann_bc_list:
            node_idx = np.argmin(np.abs(node_coords - bc['x_location']))
            F[node_idx] += bc['load_mag']
    dirichlet_map = {}
    for bc in dirichlet_bc_list:
        node_idx = np.argmin(np.abs(node_coords - bc['x_location']))
        dirichlet_map[node_idx] = bc['u_prescribed']
    dirichlet_nodes = np.array(sorted(dirichlet_map.keys()))
    u_p = np.array([dirichlet_map[i] for i in dirichlet_nodes])
    all_nodes = np.arange(n_nodes)
    free_nodes = np.setdiff1d(all_nodes, dirichlet_nodes)
    K_ff = K[np.ix_(free_nodes, free_nodes)]
    K_fp = K[np.ix_(free_nodes, dirichlet_nodes)]
    F_f = F[free_nodes]
    F_modified = F_f - K_fp @ u_p
    u_f = np.linalg.solve(K_ff, F_modified)
    displacements = np.zeros(n_nodes)
    displacements[free_nodes] = u_f
    displacements[dirichlet_nodes] = u_p
    K_pf = K[np.ix_(dirichlet_nodes, free_nodes)]
    K_pp = K[np.ix_(dirichlet_nodes, dirichlet_nodes)]
    F_p = F[dirichlet_nodes]
    reactions = K_pf @ u_f + K_pp @ u_p - F_p
    return {'displacements': displacements, 'reactions': reactions, 'node_coords': node_coords, 'reaction_nodes': dirichlet_nodes}