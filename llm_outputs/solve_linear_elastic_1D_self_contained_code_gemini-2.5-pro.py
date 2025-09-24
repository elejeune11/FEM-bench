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
    num_nodes = num_elements + 1
    node_coords = np.linspace(x_min, x_max, num_nodes)
    element_nodes = np.array([[i, i + 1] for i in range(num_elements)])
    element_E = np.zeros(num_elements)
    element_A = np.zeros(num_elements)
    for i in range(num_elements):
        elem_mid_coord = (node_coords[i] + node_coords[i + 1]) / 2.0
        found_material = False
        for region in material_regions:
            if region['coord_min'] <= elem_mid_coord <= region['coord_max']:
                element_E[i] = region['E']
                element_A[i] = region['A']
                found_material = True
                break
        if not found_material:
            raise ValueError(f'No material defined for element {i} at x={elem_mid_coord}')
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
    K = np.zeros((num_nodes, num_nodes))
    F = np.zeros(num_nodes)
    for e in range(num_elements):
        (node1_idx, node2_idx) = element_nodes[e]
        (x1, x2) = (node_coords[node1_idx], node_coords[node2_idx])
        Le = x2 - x1
        E = element_E[e]
        A = element_A[e]
        ke = E * A / Le * np.array([[1, -1], [-1, 1]])
        fe = np.zeros(2)
        J = Le / 2.0
        for i in range(n_gauss):
            xi = gauss_points[i]
            w = gauss_weights[i]
            N1_xi = (1.0 - xi) / 2.0
            N2_xi = (1.0 + xi) / 2.0
            N_xi = np.array([N1_xi, N2_xi])
            x_gauss = x1 * N1_xi + x2 * N2_xi
            fe += N_xi * body_force_fn(x_gauss) * J * w
        indices = np.array([node1_idx, node2_idx])
        K[np.ix_(indices, indices)] += ke
        F[indices] += fe
    if neumann_bc_list is not None:
        for bc in neumann_bc_list:
            x_loc = bc['x_location']
            load = bc['load_mag']
            node_idx = np.argmin(np.abs(node_coords - x_loc))
            if not np.isclose(node_coords[node_idx], x_loc):
                raise ValueError(f'Neumann BC at x={x_loc} does not match any node coordinate.')
            F[node_idx] += load
    prescribed_dofs_list = []
    prescribed_vals_list = []
    for bc in dirichlet_bc_list:
        x_loc = bc['x_location']
        u_val = bc['u_prescribed']
        node_idx = np.argmin(np.abs(node_coords - x_loc))
        if not np.isclose(node_coords[node_idx], x_loc):
            raise ValueError(f'Dirichlet BC at x={x_loc} does not match any node coordinate.')
        prescribed_dofs_list.append(node_idx)
        prescribed_vals_list.append(u_val)
    prescribed_dofs = np.array(prescribed_dofs_list, dtype=int)
    u_p = np.array(prescribed_vals_list)
    all_dofs = np.arange(num_nodes)
    free_dofs = np.setdiff1d(all_dofs, prescribed_dofs)
    K_ff = K[np.ix_(free_dofs, free_dofs)]
    K_fp = K[np.ix_(free_dofs, prescribed_dofs)]
    F_f = F[free_dofs]
    F_f_mod = F_f - K_fp @ u_p
    u_f = np.linalg.solve(K_ff, F_f_mod)
    displacements = np.zeros(num_nodes)
    displacements[free_dofs] = u_f
    displacements[prescribed_dofs] = u_p
    K_pf = K[np.ix_(prescribed_dofs, free_dofs)]
    K_pp = K[np.ix_(prescribed_dofs, prescribed_dofs)]
    F_p = F[prescribed_dofs]
    reactions = K_pf @ u_f + K_pp @ u_p - F_p
    result = {'displacements': displacements, 'reactions': reactions, 'node_coords': node_coords, 'reaction_nodes': prescribed_dofs}
    return result