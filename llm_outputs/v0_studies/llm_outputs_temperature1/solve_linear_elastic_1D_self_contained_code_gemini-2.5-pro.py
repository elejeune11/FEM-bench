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
        val = 1.0 / np.sqrt(3.0)
        gauss_points = np.array([-val, val])
        gauss_weights = np.array([1.0, 1.0])
    elif n_gauss == 3:
        val = np.sqrt(3.0 / 5.0)
        gauss_points = np.array([-val, 0.0, val])
        gauss_weights = np.array([5.0 / 9.0, 8.0 / 9.0, 5.0 / 9.0])
    else:
        raise ValueError('n_gauss must be 1, 2, or 3')
    for e in range(num_elements):
        node_indices = np.array([e, e + 1])
        (x_a, x_b) = node_coords[node_indices]
        L_e = x_b - x_a
        elem_mid_coord = (x_a + x_b) / 2.0
        (E, A) = (None, None)
        for region in material_regions:
            if region['coord_min'] <= elem_mid_coord < region['coord_max'] or (e == num_elements - 1 and region['coord_min'] <= elem_mid_coord <= region['coord_max']):
                E = region['E']
                A = region['A']
                break
        if E is None or A is None:
            raise ValueError(f'No material properties found for element {e} at x={elem_mid_coord}')
        k_e = E * A / L_e * np.array([[1, -1], [-1, 1]])
        f_e = np.zeros(2)
        J = L_e / 2.0
        for (xi, w) in zip(gauss_points, gauss_weights):
            N = np.array([(1 - xi) / 2.0, (1 + xi) / 2.0])
            x_xi = x_a * N[0] + x_b * N[1]
            f_e += N * body_force_fn(x_xi) * A * w * J
        ix = np.ix_(node_indices, node_indices)
        K[ix] += k_e
        F[node_indices] += f_e
    if neumann_bc_list:
        for bc in neumann_bc_list:
            x_loc = bc['x_location']
            load = bc['load_mag']
            node_idx = np.argmin(np.abs(node_coords - x_loc))
            F[node_idx] += load
    prescribed_dofs_map = {}
    for bc in dirichlet_bc_list:
        x_loc = bc['x_location']
        u_val = bc['u_prescribed']
        node_idx = np.argmin(np.abs(node_coords - x_loc))
        prescribed_dofs_map[node_idx] = u_val
    prescribed_dofs = np.array(sorted(prescribed_dofs_map.keys()), dtype=int)
    prescribed_vals = np.array([prescribed_dofs_map[i] for i in prescribed_dofs])
    all_dofs = np.arange(n_nodes)
    free_dofs = np.setdiff1d(all_dofs, prescribed_dofs)
    K_ff = K[np.ix_(free_dofs, free_dofs)]
    K_fp = K[np.ix_(free_dofs, prescribed_dofs)]
    F_f = F[free_dofs]
    F_mod = F_f - K_fp @ prescribed_vals
    u_f = np.linalg.solve(K_ff, F_mod)
    displacements = np.zeros(n_nodes)
    displacements[free_dofs] = u_f
    displacements[prescribed_dofs] = prescribed_vals
    full_ku_product = K @ displacements
    reactions = full_ku_product[prescribed_dofs] - F[prescribed_dofs]
    return {'displacements': displacements, 'reactions': reactions, 'node_coords': node_coords, 'reaction_nodes': prescribed_dofs}